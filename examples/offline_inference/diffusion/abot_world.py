# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run ABot-World offline image-to-video generation.

Usage (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
    DIFFUSERS_OFFLINE=1 python examples/offline_inference/diffusion/abot_world.py \\
      --model acvlab/ABot-World-0-5B-LF \\
      --image /path/to/first_frame.png \\
      --prompt "The camera moves forward through the scene." \\
      --num-frames 9 \\
      --output abot_world_output.mp4
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_MODEL = "acvlab/ABot-World-0-5B-LF"
_VAE_DIT_SPATIAL_FACTOR = 32
_PAGED_KV_BLOCK_ALIGNMENT = 16


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABot-World offline image-to-video generation.")
    parser.add_argument("--model", default=_MODEL, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--image", required=True, help="Path to the initial RGB image.")
    parser.add_argument("--prompt", required=True, help="Scene description prompt.")
    parser.add_argument("--num-frames", type=int, default=9, help="Number of pixel frames (9+12k, at most 117).")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output video file path (MP4).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--flow-shift", type=float, default=5.0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    image = Path(args.image).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not image.is_file():
        raise ValueError("--image must point to an existing file.")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be positive multiples of 32.")
    tokens_per_frame = (
        args.height // _VAE_DIT_SPATIAL_FACTOR
    ) * (args.width // _VAE_DIT_SPATIAL_FACTOR)
    if tokens_per_frame % _PAGED_KV_BLOCK_ALIGNMENT:
        raise ValueError(
            "FlashAttention paged KV requires tokens per frame to be a multiple "
            f"of 16; got {tokens_per_frame}. Use --height 512 --width 832."
        )
    if args.num_frames < 9 or args.num_frames > 117 or (args.num_frames - 9) % 12:
        raise ValueError("--num-frames must follow 9+12k and be at most 117 (9, 21, ..., 117).")
    if not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text.")
    return image, output


async def run(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    image, output_path = _validate_args(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = AsyncOmni(
        model=args.model,
        model_class_name="ABotWorldCausalPipeline",
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        vae_use_tiling=args.vae_use_tiling,
        parallel_config={"tensor_parallel_size": args.tensor_parallel_size},
        max_num_seqs=1,
    )
    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=args.seed,
        output_type="np",
        extra_args={"flow_shift": args.flow_shift},
    )
    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": str(image)},
    }

    video = None
    try:
        outputs = engine.generate(prompt, sampling_params_list=[sampling])
        async for output in outputs:
            if output.finished and output.stage_id == 0:
                if output.error:
                    raise RuntimeError(f"Generation failed: {output.error}")
                video_output = output.multimodal_output
                if isinstance(video_output, dict):
                    video = (video_output.get("payload") or {}).get("video")
                break
    finally:
        engine.shutdown()

    if video is None:
        raise RuntimeError("Generation finished without a video payload.")

    import numpy as np
    from diffusers.utils import export_to_video

    frames = np.asarray(video)
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected generated video shape: {frames.shape}.")
    export_to_video(frames, str(output_path), fps=16)
    print(f"Video saved to {output_path}")
    return output_path


def main(argv: Sequence[str] | None = None) -> Path:
    import asyncio
    return asyncio.run(run(argv))


if __name__ == "__main__":
    main()
