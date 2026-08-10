# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run ABot-World offline image-to-video generation.

Usage:
    python examples/offline_inference/diffusion/abot_world.py \\
      --model acvlab/ABot-World-0-5B-LF \\
      --image /path/to/first_frame.png \\
      --prompt "The camera moves forward through the scene." \\
      --num-frames 31 \\
      --output abot_world_output.mp4
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

_MODEL = "acvlab/ABot-World-0-5B-LF"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABot-World offline image-to-video generation.")
    parser.add_argument("--model", default=_MODEL, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--image", required=True, help="Path to the initial RGB image.")
    parser.add_argument("--prompt", required=True, help="Scene description prompt.")
    parser.add_argument("--num-frames", type=int, default=31, help="Number of pixel frames to generate (9+12k pattern).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output video file path (MP4).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--flow-shift", type=float, default=5.0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    image = Path(args.image).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not image.is_file():
        raise ValueError("--image must point to an existing file.")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.num_frames <= 0 or (args.num_frames - 1) % 4:
        raise ValueError("--num-frames must be (4k + 1) for causal VAE geometry (e.g. 9, 13, 31, 81, 121).")
    if not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text.")
    return image, output


def run(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    image, output_path = _validate_args(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = AsyncOmni(
        model=args.model,
        enforce_eager=args.enforce_eager,
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
        output_type="mp4",
        extra_args={"flow_shift": args.flow_shift},
    )
    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": str(image)},
    }

    outputs = engine.generate(prompt, sampling_params_list=[sampling])
    for output in outputs:
        if output.finished and output.stage_id == 0:
            if output.error:
                raise RuntimeError(f"Generation failed: {output.error}")
            video_output = output.multimodal_output
            if isinstance(video_output, dict) and "payload" in video_output:
                video = video_output["payload"].get("video")
                if isinstance(video, torch.Tensor):
                    from diffusers.video_processor import VideoProcessor
                    vp = VideoProcessor(vae_scale_factor=8)
                    video_np = vp.postprocess_video(video, output_type="np")
                    import numpy as np
                    import imageio
                    frames_uint8 = [(frame * 255).astype(np.uint8) for frame in video_np[0]]
                    writer = imageio.get_writer(str(output_path), fps=16)
                    for frame in frames_uint8:
                        writer.append_data(frame)
                    writer.close()
                    print(f"Video saved to {output_path}")
                    break
                elif isinstance(video, (list, tuple)):
                    # Already post-processed
                    import numpy as np
                    import imageio
                    frames_uint8 = []
                    for item in video:
                        if isinstance(item, torch.Tensor):
                            item = item.cpu().numpy()
                        item = np.asarray(item)
                        if item.max() <= 1.0:
                            item = (item * 255).astype(np.uint8)
                        frames_uint8.append(item)
                    writer = imageio.get_writer(str(output_path), fps=16)
                    for frame in frames_uint8:
                        writer.append_data(frame)
                    writer.close()
                    print(f"Video saved to {output_path}")
                    break

    engine.shutdown()
    return output_path


def main(argv: Sequence[str] | None = None) -> Path:
    return run(argv)


if __name__ == "__main__":
    main()
