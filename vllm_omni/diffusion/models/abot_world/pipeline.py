# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ABot-World causal DMD pipeline with offline and realtime inference."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import retrieve_latents
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionTickRequest,
)

from .transformer import (
    ABotAttentionCache,
    ABotTransformerCache,
    ABotWorldCausalTransformer3DModel,
    allocate_abot_cache,
)

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmProgressBar

    from vllm_omni.experimental.ar_diffusion.kv_cache.state import (
        ARDiffusionKVState,
    )

ABOT_DMD_TIMESTEPS = (1000, 750, 500, 250)
_WINDOW_FRAMES = 21
_MAX_RAW_FRAMES = 121
_MAX_SEQUENCE_LENGTH = 512
_REFERENCE_RESOLUTION = 512
_MAX_SOURCE_IMAGE_PIXELS = 4096 * 4096
_DEFAULT_HEIGHT = 480
_DEFAULT_WIDTH = 832
_SOURCE_IMAGE_ERROR = (
    "Unable to load multi_modal_data.image; expected a decodable image within 4096 * 4096 source pixels."
)
_PREPROCESSED_ACTION_KEY = "_abot_camera_actions"
_ACTION_CONTROL_DIM = 32


@dataclass(frozen=True)
class _ABotRequestInputs:
    prompt: str
    image: PIL.Image.Image | torch.Tensor
    reference_images: tuple[PIL.Image.Image | torch.Tensor, ...] | None
    camera_actions: tuple[tuple[str, ...], ...] | None
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    num_frame_per_block: int
    output_type: str
    max_sequence_length: int
    flow_shift: float
    generator: torch.Generator


@dataclass
class _ABotARSessionState:
    next_chunk_index: int = 0
    prompt: str | None = None
    generator_state: torch.Tensor | None = None
    first_frame_latent: torch.Tensor | None = None
    current_actions: tuple[tuple[str, ...], ...] | None = None


def _positive_finite_flow_shift(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("flow_shift must be a positive finite number.")
    try:
        flow_shift = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("flow_shift must be a positive finite number.") from exc
    if not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("flow_shift must be a positive finite number.")
    return flow_shift


def _build_shifted_flow_schedule(*, flow_shift: float) -> tuple[tuple[float, float], ...]:
    base_sigmas = torch.tensor(ABOT_DMD_TIMESTEPS, dtype=torch.float64) / 1000
    shifted_numerators = flow_shift * base_sigmas
    shifted_sigmas = shifted_numerators / ((1.0 - base_sigmas) + shifted_numerators)
    warped_timesteps = shifted_sigmas * 1000
    return tuple(
        (float(ts), float(sigma))
        for ts, sigma in zip(warped_timesteps.tolist(), shifted_sigmas.tolist(), strict=True)
    )


def _validate_parallel_config(od_config: OmniDiffusionConfig) -> None:
    if getattr(od_config, "quantization_config", None) is not None:
        raise NotImplementedError("ABot-World does not support quantization.")
    parallel_config = getattr(od_config, "parallel_config", None)
    if parallel_config is None:
        return
    unsupported = {
        "pipeline_parallel_size": "pipeline parallelism",
        "sequence_parallel_size": "sequence parallelism",
        "cfg_parallel_size": "CFG parallelism",
        "vae_patch_parallel_size": "VAE parallelism",
    }
    for field, feature in unsupported.items():
        size = getattr(parallel_config, field, 1) or 1
        if size > 1:
            raise NotImplementedError(f"ABot-World does not support {feature} ({field}={size}).")


def _decode_source_image(image: PIL.Image.Image) -> PIL.Image.Image:
    try:
        return image.convert("RGB")
    except (OSError, SyntaxError, ValueError, PIL.Image.DecompressionBombError):
        raise ValueError(_SOURCE_IMAGE_ERROR) from None


def _load_source_image(path: str | os.PathLike[str]) -> PIL.Image.Image:
    try:
        source_image = PIL.Image.open(path)
    except (OSError, SyntaxError, ValueError, PIL.Image.DecompressionBombError):
        raise ValueError(_SOURCE_IMAGE_ERROR) from None
    try:
        return _decode_source_image(source_image)
    finally:
        source_image.close()


def get_abot_world_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    del od_config

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if not isinstance(prompt, dict):
            raise ValueError("ABot-World requires a prompt mapping containing multi_modal_data.image.")
        multi_modal_data = prompt.get("multi_modal_data") or {}
        if not isinstance(multi_modal_data, dict):
            raise ValueError("prompt.multi_modal_data must be a mapping.")

        image = multi_modal_data.get("image")
        if image is None:
            raise ValueError("ABot-World requires exactly one image in multi_modal_data.image.")
        if isinstance(image, (str, os.PathLike)):
            image = _load_source_image(image)
        elif isinstance(image, PIL.Image.Image):
            image = _decode_source_image(image)
        elif not isinstance(image, torch.Tensor):
            raise ValueError("multi_modal_data.image must be a PIL image, tensor, or file path.")

        reference_images = multi_modal_data.get("reference_images")
        resolved_refs: list[PIL.Image.Image | torch.Tensor] = []
        if reference_images is not None:
            if not isinstance(reference_images, (list, tuple)):
                raise ValueError("multi_modal_data.reference_images must be a list.")
            for ref in reference_images:
                if isinstance(ref, (str, os.PathLike)):
                    resolved_refs.append(_load_source_image(ref))
                elif isinstance(ref, PIL.Image.Image):
                    resolved_refs.append(_decode_source_image(ref))
                elif isinstance(ref, torch.Tensor):
                    resolved_refs.append(ref)
                else:
                    raise ValueError("Each reference image must be a PIL image, tensor, or file path.")

        tick = ARDiffusionTickRequest.from_extra_args(
            getattr(request.sampling_params, "extra_args", None) or {}
        )
        camera_actions = None
        if tick is not None:
            camera_controls = [c for c in tick.controls if c.track == "camera"]
            if camera_controls:
                from vllm_omni.diffusion.models.abot_world.actions import (
                    ABOT_CAMERA_ACTION_SCHEMA,
                    parse_abot_camera_action_frames,
                )
                if camera_controls[0].schema == ABOT_CAMERA_ACTION_SCHEMA:
                    camera_actions = parse_abot_camera_action_frames(camera_controls[0].data, expected_frames=3)

        extra_args = dict(getattr(request.sampling_params, "extra_args", None) or {})
        updated_prompt = dict(prompt)
        updated_mmd = dict(multi_modal_data)
        updated_mmd["image"] = image
        if resolved_refs:
            updated_mmd["reference_images"] = tuple(resolved_refs)
        updated_prompt["multi_modal_data"] = updated_mmd
        request.prompt = updated_prompt
        extra_args[_PREPROCESSED_ACTION_KEY] = camera_actions
        request.sampling_params.extra_args = extra_args
        return request

    return pre_process_func


def get_abot_world_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., Any]:
    del od_config
    from diffusers.video_processor import VideoProcessor
    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video, output_type="np", sampling_params=None):
        if isinstance(video, dict) and isinstance(video.get("payload"), dict):
            return video
        if sampling_params is not None:
            output_type = getattr(sampling_params, "output_type", None) or output_type
        if output_type == "latent":
            return video
        return {
            "payload": {"video": video_processor.postprocess_video(video, output_type=output_type)},
            "metadata": {},
        }
    return post_process_func


class ABotWorldCausalPipeline(
    nn.Module,
    SupportImageInput,
    SupportsComponentDiscovery,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """ABot-World I2V causal DMD pipeline."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    dummy_run_num_frames: ClassVar[int] = 0
    _AR_BRANCH = "main"
    _AR_TEXT_CACHE = "text"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        _validate_parallel_config(od_config)
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)
        managed_offload = bool(
            getattr(od_config, "enable_cpu_offload", False)
            or getattr(od_config, "enable_layerwise_offload", False)
        )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model, subfolder="", revision=None,
                prefix="", fall_back_to_pt=True,
            )
        ]

        # Tokenizer from bundled google/umt5-xxl (no HF fallback)
        tokenizer_path = os.path.join(model, "google", "umt5-xxl")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        # Text encoder
        self.text_encoder = self._load_text_encoder(model, dtype, local_files_only)
        if not managed_offload:
            self.text_encoder = self.text_encoder.to(self.device)

        # VAE
        self.vae = self._load_vae(model, dtype, local_files_only)
        if not managed_offload:
            self.vae = self.vae.to(self.device)

        # Custom causal transformer
        self.transformer = self._create_transformer(model, local_files_only)

        self.vae_scale_factor_temporal = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        self.vae_scale_factor_spatial = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        self._num_frame_per_block = 3

        model_config = getattr(od_config, "model_config", None) or {}
        self._ar_height = int(model_config.get("ar_diffusion_height", _DEFAULT_HEIGHT))
        self._ar_width = int(model_config.get("ar_diffusion_width", _DEFAULT_WIDTH))
        self._ar_diffusion_kv_state: ARDiffusionKVState | None = None
        self._ar_sessions: dict[str, _ABotARSessionState] = {}

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["vae.encode", "vae.decode", "_generate_block",
                            "text_encoder.forward", "tokenizer.forward"],
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def _load_text_encoder(self, model: str, dtype: torch.dtype, local_files_only: bool) -> UMT5EncoderModel:
        from transformers import T5Config

        tokenizer_path = os.path.join(model, "google", "umt5-xxl")
        encoder_pth = os.path.join(model, "models_t5_umt5-xxl-enc-bf16.pth")

        # UMT5-XXL encoder-only config (is_encoder_decoder=False is critical
        # to avoid causal mask computation in the encoder).
        config = T5Config(
            d_model=4096, d_kv=64, d_ff=10240, num_layers=24,
            num_decoder_layers=0, num_heads=64, relative_attention_num_buckets=32,
            relative_attention_max_distance=128, dropout_rate=0.0,
            layer_norm_epsilon=1e-06, feed_forward_proj="gated-gelu",
            is_encoder_decoder=False,
            pad_token_id=0, vocab_size=256384,
        )

        text_encoder = UMT5EncoderModel(config)
        if os.path.isfile(encoder_pth):
            state_dict = torch.load(encoder_pth, map_location="cpu", weights_only=True)
            text_encoder.load_state_dict(state_dict, strict=False)
        return text_encoder

    def _load_vae(self, model: str, dtype: torch.dtype, local_files_only: bool) -> DistributedAutoencoderKLWan:
        # Load Wan2.2 VAE from bundled .pth (no HF fallback).
        vae_pth = os.path.join(model, "Wan2.2_VAE.pth")
        if os.path.isfile(vae_pth):
            return self._load_vae_from_local_pth(vae_pth, dtype)
        raise FileNotFoundError(f"Wan2.2_VAE.pth not found at {vae_pth}")

    @staticmethod
    def _load_vae_from_local_pth(vae_pth: str, dtype: torch.dtype) -> DistributedAutoencoderKLWan:
        """Load Wan2.2 VAE from a standalone .pth checkpoint by constructing
        a minimal compatible config inline."""
        from diffusers.models.autoencoders import AutoencoderKLWan

        # Wan2.2 VAE config (matches the checkpoint bundled with ABot-World).
        vae_config = {
            "_class_name": "AutoencoderKLWan",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 48,
            "down_block_types": ("WanResnetDownsampleBlock3D",) * 4,
            "up_block_types": ("WanResnetUpsampleBlock3D",) * 4,
            "block_out_channels": (128, 256, 512, 512),
            "layers_per_block": 2,
            "act_fn": "silu",
            "scaling_factor": 0.4769,
            "latents_mean": None,
            "latents_std": None,
            "norm_num_groups": 32,
            "shift_factor": None,
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 8,
            "use_parallel_blocks": True,
            "mid_block_add_conv": True,
        }
        vae = DistributedAutoencoderKLWan.from_config(vae_config)
        state_dict = torch.load(vae_pth, map_location="cpu", weights_only=True)
        vae.load_state_dict(state_dict, strict=False)
        vae = vae.to(dtype=dtype)
        vae.init_distributed()
        return vae

    def _create_transformer(self, model: str, local_files_only: bool) -> ABotWorldCausalTransformer3DModel:
        import json
        config_path = os.path.join(model, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                raw_config = json.load(f)
        else:
            raise FileNotFoundError(f"ABot-World config.json not found at {config_path}")
        return ABotWorldCausalTransformer3DModel.from_config(raw_config)

    # ── SupportsARDiffusionPipeline ──────────────────────────────────────

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        cfg = self.transformer.config
        patch_h, patch_w = cfg.patch_size[1], cfg.patch_size[2]
        spatial = self.vae_scale_factor_spatial
        latent_h, latent_w = self._ar_height // spatial, self._ar_width // spatial
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        tp_size = get_tensor_model_parallel_world_size()
        num_local_heads = int(cfg.num_attention_heads) // tp_size

        horizon = (_MAX_RAW_FRAMES - 1) // self.vae_scale_factor_temporal + 1
        condition_bytes = (
            int(cfg.out_channels) * horizon * latent_h * latent_w
            * torch.empty((), dtype=self.transformer.dtype).element_size()
        )
        return ARDiffusionKVCacheSpec(
            num_layers=int(cfg.num_layers),
            num_kv_heads=num_local_heads,
            head_size=int(cfg.attention_head_dim),
            tokens_per_frame=tokens_per_frame,
            frames_per_block=self._num_frame_per_block,
            window_frames=_WINDOW_FRAMES,
            sink_frames=0,
            kv_branches=(ARDiffusionKVBranchSpec(self._AR_BRANCH, 0),),
            session_capacity=2,
            cross_attention=(ARDiffusionCrossAttentionKVSpec(self._AR_TEXT_CACHE, _MAX_SEQUENCE_LENGTH),),
            model_owned_state_bytes_per_session=condition_bytes,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("ABot AR-Diffusion state is already bound.")
        if state.session_id != session_id:
            raise ValueError(f"Session mismatch: {state.session_id!r} != {session_id!r}.")
        self._ar_diffusion_kv_state = state
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)

    # ── Request parsing ──────────────────────────────────────────────────

    def _parse_request(self, req: DiffusionRequestBatch) -> _ABotRequestInputs:
        if req.num_reqs != 1:
            raise ValueError("ABot-World supports a single prompt request.")
        sampling = req.sampling_params
        generator = getattr(sampling, "generator", None)
        if not isinstance(generator, torch.Generator):
            raise ValueError("ABot-World requires the runner-provided torch.Generator.")

        prompt_value = req.prompts[0]
        multi_modal_data: dict[str, Any]
        if isinstance(prompt_value, dict):
            prompt = prompt_value.get("prompt") or ""
            multi_modal_data = prompt_value.get("multi_modal_data") or {}
        else:
            prompt = str(prompt_value)
            multi_modal_data = {}

        image = multi_modal_data.get("image")
        if isinstance(image, (str, os.PathLike)):
            raise ValueError("file-path images must be materialized by pre-process.")
        if not isinstance(image, (PIL.Image.Image, torch.Tensor)):
            raise ValueError("multi_modal_data.image must be a PIL image or tensor.")

        ref_raw = multi_modal_data.get("reference_images")
        reference_images: tuple[PIL.Image.Image | torch.Tensor, ...] | None = None
        if ref_raw is not None:
            resolved = []
            for r in ref_raw:
                if isinstance(r, (str, os.PathLike)):
                    raise ValueError("file-path reference images must be materialized by pre-process.")
                resolved.append(r)
            reference_images = tuple(resolved)

        extra_args = getattr(sampling, "extra_args", None) or {}
        camera_actions = extra_args.get(_PREPROCESSED_ACTION_KEY)
        flow_shift = _positive_finite_flow_shift(extra_args.get("flow_shift", 5.0))

        height = getattr(sampling, "height", None) or _DEFAULT_HEIGHT
        width = getattr(sampling, "width", None) or _DEFAULT_WIDTH
        patch_size = tuple(self.transformer.config.patch_size)
        div = self.vae_scale_factor_spatial * patch_size[1]
        if height % div or width % div:
            raise ValueError(f"height/width must be divisible by {div}.")

        num_frames = getattr(sampling, "num_frames", None) or 31
        if (num_frames - 1) % self.vae_scale_factor_temporal:
            raise ValueError(f"(num_frames - 1) must be divisible by {self.vae_scale_factor_temporal}.")
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        if num_latent_frames % self._num_frame_per_block:
            raise ValueError(f"num_latent_frames ({num_latent_frames}) must be divisible by {self._num_frame_per_block}.")

        return _ABotRequestInputs(
            prompt=prompt.strip(),
            image=image,
            reference_images=reference_images,
            camera_actions=camera_actions,
            height=height, width=width,
            num_frames=num_frames,
            num_latent_frames=num_latent_frames,
            num_frame_per_block=self._num_frame_per_block,
            output_type=getattr(sampling, "output_type", None) or "np",
            max_sequence_length=getattr(sampling, "max_sequence_length", None) or _MAX_SEQUENCE_LENGTH,
            flow_shift=flow_shift,
            generator=generator,
        )

    # ── Image / VAE helpers ──────────────────────────────────────────────

    def _prepare_image_tensor(self, image: PIL.Image.Image | torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        if isinstance(image, PIL.Image.Image):
            arr = np.asarray(image, dtype=np.float32).copy()
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 255.0
        else:
            t = image.detach()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            t = t.to(dtype=torch.float32)
            if t.max() > 1.0:
                t = t / 255.0
        if t.min() >= 0.0:
            t = t * 2.0 - 1.0
        if t.shape[-2:] != (height, width):
            t = F.interpolate(t, size=(height, width), mode="bicubic", align_corners=False)
        return t.to(device=self.device, dtype=torch.float32)

    def _vae_latent_stats(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, -1, 1, 1, 1)
        # Read from buffer (populated by load_state_dict), fall back to config
        mean_val = getattr(self.vae, "latents_mean", None)
        if mean_val is None:
            mean_val = self.vae.config.latents_mean
        std_val = getattr(self.vae, "latents_std", None)
        if std_val is None:
            std_val = self.vae.config.latents_std
        if mean_val is None or std_val is None:
            return torch.as_tensor(0.0, device=ref.device, dtype=ref.dtype), torch.as_tensor(1.0, device=ref.device, dtype=ref.dtype)
        mean = torch.as_tensor(mean_val, device=ref.device, dtype=ref.dtype).view(*shape)
        std = torch.as_tensor(std_val, device=ref.device, dtype=ref.dtype).view(*shape)
        return mean, std

    def _encode_first_frame(self, image: PIL.Image.Image | torch.Tensor, height: int, width: int,
                            dtype: torch.dtype) -> torch.Tensor:
        """VAE-encode the first frame image into a single 48-channel latent frame."""
        img = self._prepare_image_tensor(image, height=height, width=width)
        video = img.unsqueeze(2)  # [1, 3, 1, H, W]
        latent = retrieve_latents(self.vae.encode(video.to(dtype=self.vae.dtype)), sample_mode="argmax")
        mean, std = self._vae_latent_stats(latent)
        return ((latent - mean) / std).to(dtype=dtype)

    def _build_action_tensor(self, camera_actions: tuple[tuple[str, ...], ...] | None,
                              num_latent_frames: int, height: int, width: int,
                              dtype: torch.dtype) -> torch.Tensor:
        """Build [1, 32, num_latent_frames, height, width] action tensor."""
        action_keys = ("w", "a", "s", "d", "i", "j", "k", "l")
        t = torch.zeros(1, num_latent_frames, len(action_keys), device=self.device, dtype=dtype)
        if camera_actions is not None:
            for fi, actions in enumerate(camera_actions):
                if fi >= num_latent_frames:
                    break
                for a in actions:
                    if a in action_keys:
                        t[0, fi, action_keys.index(a)] = 1.0
        # [1, F, 8] → [1, F, 1, 1, 8] → broadcast to [1, F, height, width, 8]
        t = t.unsqueeze(2).unsqueeze(2).expand(-1, -1, height, width, -1)
        # Permute to [1, 8, F, H, W], then repeat_interleave to get 32 channels
        t = t.permute(0, 4, 1, 2, 3).contiguous()  # [1, 8, F, H, W]
        t = t.repeat_interleave(4, dim=1)  # [1, 32, F, H, W]
        return t.to(dtype=dtype)

    def encode_prompt(self, prompt: str, *, max_sequence_length: int, dtype: torch.dtype) -> torch.Tensor:
        text_inputs = self.tokenizer(
            [" ".join(prompt.strip().split())],
            padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        embeds = self.text_encoder(input_ids, attention_mask).last_hidden_state
        embeds = embeds.to(device=self.device, dtype=dtype)
        return embeds * attention_mask.unsqueeze(-1).to(dtype=embeds.dtype)

    # ── Generation ───────────────────────────────────────────────────────

    def _generate_block(
        self,
        *,
        noise_latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        first_frame_latent: torch.Tensor | None,
        action_condition: torch.Tensor | None,
        start_frame: int,
        schedule: tuple[tuple[float, float], ...],
        generator: torch.Generator,
        cache: ABotTransformerCache,
        progress_bar: TqdmProgressBar[Any],
    ) -> torch.Tensor:
        out_channels = int(self.transformer.config.out_channels)
        current_latents = noise_latent
        replace_first = start_frame == 0 and first_frame_latent is not None

        for step_idx, (ts_val, sigma) in enumerate(schedule):
            if not self.od_config.enforce_eager:
                torch.compiler.cudagraph_mark_step_begin()
            set_forward_context_denoise_step_idx(step_idx)
            timestep = torch.full((1,), float(ts_val), device=self.device, dtype=torch.float32)

            # Replace first frame with clean latent at every DMD step
            if replace_first:
                current_latents[:, :, 0] = first_frame_latent[:, :, 0]

            flow_pred = self.transformer(
                hidden_states=current_latents.to(dtype=self.transformer.dtype),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                cache=cache,
                start_frame=start_frame,
                update_cache=False,
                action_condition=action_condition,
            )
            flow_pred = flow_pred[:, :out_channels]

            x0 = current_latents - sigma * flow_pred.float()
            if step_idx + 1 < len(schedule):
                next_sigma = schedule[step_idx + 1][1]
                noise = randn_tensor(current_latents.shape, generator=generator,
                                     device=self.device, dtype=torch.float32)
                current_latents = (1.0 - next_sigma) * x0 + next_sigma * noise
            else:
                current_latents = x0
            progress_bar.update()

        # Commit clean latents to KV cache (context_noise=0)
        _ = self.transformer(
            hidden_states=current_latents.to(dtype=self.transformer.dtype),
            timestep=torch.zeros(1, device=self.device, dtype=torch.float32),
            encoder_hidden_states=prompt_embeds,
            cache=cache,
            start_frame=start_frame,
            update_cache=True,
            action_condition=action_condition,
        )
        return current_latents

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        inputs = self._parse_request(req)
        tick = ARDiffusionTickRequest.from_extra_args(req.sampling_params.extra_args)
        if tick is not None and self._ar_diffusion_kv_state is None:
            raise RuntimeError("ABot typed ticks require ARDiffusionEngine session binding.")
        if tick is None and self._ar_diffusion_kv_state is not None:
            raise ValueError("ABot ARDiffusionEngine requests must carry ar_diffusion_tick.")

        session_state: _ABotARSessionState | None = None
        if tick is not None:
            if (inputs.height, inputs.width) != (self._ar_height, self._ar_width):
                raise ValueError("ABot AR-Diffusion resolution must match fixed cache geometry.")
            session_state = self._ar_sessions.setdefault(tick.session_id, _ABotARSessionState())

        schedule = _build_shifted_flow_schedule(flow_shift=inputs.flow_shift)
        dtype = self.transformer.dtype
        out_channels = int(self.transformer.config.out_channels)
        block_frames = inputs.num_frame_per_block

        prompt_embeds = self.encode_prompt(inputs.prompt, max_sequence_length=inputs.max_sequence_length, dtype=dtype)
        latent_h = inputs.height // self.vae_scale_factor_spatial
        latent_w = inputs.width // self.vae_scale_factor_spatial

        if tick is None:
            # ── Offline: full video generation ──
            ff_latent = self._encode_first_frame(
                inputs.image, inputs.height, inputs.width, dtype,
            )
            action_cond = self._build_action_tensor(
                inputs.camera_actions, inputs.num_latent_frames,
                inputs.height, inputs.width, dtype,
            )
            cache = self.transformer.allocate_cache(
                batch_size=1, latent_height=latent_h, latent_width=latent_w,
                device=self.device, dtype=dtype,
            )
            total_steps = (inputs.num_latent_frames // block_frames) * len(ABOT_DMD_TIMESTEPS)
            generated_blocks: list[torch.Tensor] = []

            with self.progress_bar(total=total_steps) as progress_bar:
                for local_start in range(0, inputs.num_latent_frames, block_frames):
                    stop = local_start + block_frames
                    # Slice action condition for this block
                    block_action = action_cond[:, :, local_start:stop] if action_cond is not None else None
                    # First frame only applies to the very first block
                    block_ff = ff_latent if local_start == 0 else None

                    _, _, _, latent_h_block, latent_w_block = ff_latent.shape
                    noise = randn_tensor(
                        (1, out_channels, block_frames, latent_h_block, latent_w_block),
                        generator=inputs.generator, device=self.device, dtype=torch.float32,
                    )
                    generated_blocks.append(self._generate_block(
                        noise_latent=noise, prompt_embeds=prompt_embeds,
                        first_frame_latent=block_ff, action_condition=block_action,
                        start_frame=local_start, schedule=schedule,
                        generator=inputs.generator, cache=cache,
                        progress_bar=progress_bar,
                    ))
        else:
            # ── Realtime: single block ──
            assert session_state is not None
            if session_state.first_frame_latent is None:
                session_state.first_frame_latent = self._encode_first_frame(
                    inputs.image, inputs.height, inputs.width, dtype,
                )
            # First frame only applies to the very first chunk
            ff_latent = session_state.first_frame_latent if tick.chunk_index == 0 else None

            _, _, _, latent_h_block, latent_w_block = session_state.first_frame_latent.shape
            block_action = self._build_action_tensor(
                inputs.camera_actions, block_frames, inputs.height, inputs.width, dtype,
            )
            if session_state.generator_state is not None:
                inputs.generator.set_state(session_state.generator_state)

            noise = randn_tensor(
                (1, out_channels, block_frames, latent_h_block, latent_w_block),
                generator=inputs.generator, device=self.device, dtype=torch.float32,
            )
            cache = self.transformer.allocate_cache(
                batch_size=1, latent_height=latent_h, latent_width=latent_w,
                device=self.device, dtype=dtype,
            )
            total_steps = len(ABOT_DMD_TIMESTEPS)
            cond_start = tick.chunk_index * block_frames
            with self.progress_bar(total=total_steps) as progress_bar:
                generated_blocks = [self._generate_block(
                    noise_latent=noise, prompt_embeds=prompt_embeds,
                    first_frame_latent=ff_latent, action_condition=block_action,
                    start_frame=cond_start, schedule=schedule,
                    generator=inputs.generator, cache=cache,
                    progress_bar=progress_bar,
                )]

        generated_latents = torch.cat(generated_blocks, dim=2)

        if tick is not None:
            assert session_state is not None
            session_state.prompt = inputs.prompt
            session_state.generator_state = inputs.generator.get_state()
            session_state.current_actions = inputs.camera_actions
            session_state.next_chunk_index += 1

            output = {
                "payload": {"latents": generated_latents},
                "metadata": {"ar_diffusion": ARDiffusionChunkMetadata.from_tick(tick).to_dict()},
            }
        elif inputs.output_type == "latent":
            output = generated_latents
        else:
            mean, std = self._vae_latent_stats(generated_latents)
            vae_latents = (generated_latents * std + mean).to(dtype=self.vae.dtype)
            output = self.vae.decode(vae_latents, return_dict=False)[0]
            if output.shape[2] != inputs.num_frames:
                raise RuntimeError(f"VAE decode mismatch: expected {inputs.num_frames} frames, got {output.shape[2]}.")

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Forward transformer-relevant weights to the transformer."""
        transformer_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name.startswith("model."):
                transformer_weights.append((name, tensor))

        self.transformer.load_weights(transformer_weights)
        # Transformer loaded from safetensors; VAE and text_encoder were
        # loaded separately in __init__. Report all parameters as loaded.
        return {name for name, _ in self.named_parameters()}
