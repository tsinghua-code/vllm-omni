# ABot-World

> Experimental: offline and realtime interactive world generation

## Summary

- Vendor: ACVLab (Amap CV Lab)
- Model: `acvlab/ABot-World-0-5B-LF`
- Task: image-conditioned interactive world generation
- Modes: offline trajectory replay and in-process realtime AR-Diffusion ticks
- Hardware validated: NVIDIA H200 and B200
- Maintainer: Community

The checkpoint is Apache-2.0 licensed. The vLLM-Omni integration code is also Apache-2.0.

## Offline generation

The offline path consumes one source image and generates a video:

```bash
python examples/offline_inference/diffusion/abot_world.py \
  --model acvlab/ABot-World-0-5B-LF \
  --image /path/to/first_frame.png \
  --prompt "The camera moves slowly forward through the scene." \
  --num-frames 31 \
  --output abot_world_output.mp4
```

Raw frame counts must be `9 + 12k` (e.g., 9, 21, 33, 81, 117, 121), up to 121 frames.

## Realtime in-process generation

Each JSONL line describes the prompt and/or three latent-frame camera
actions (W/A/S/D/I/J/K/L) applied at the next chunk boundary:

```json
{"event_id":1,"prompt":"A road through a forest","frames":[["j"],[],[]]}
{"event_id":2,"frames":[["w"],["w"],["w"]]}
{"event_id":3,"prompt":"The road enters a snowy valley","frames":[[],[],[]]}
```

Run:

```bash
python examples/offline_inference/diffusion/abot_world_realtime.py \
  --model acvlab/ABot-World-0-5B-LF \
  --image /path/to/first_frame.png \
  --prompt "Initial scene" \
  --events /path/to/events.jsonl \
  --output-dir /tmp/abot-realtime \
  --gpu-memory-fraction 0.6
```

## Current limitations

- Only the 0.5B-LF causal student checkpoint is supported.
- The realtime control plane is internal; there is no public server transport yet.
- AR-Diffusion stages require one replica.
- One AR block is generated per request and `max_num_seqs` must be one.
- SP/USP, pipeline/CFG parallelism, HSDP, VAE parallelism, quantization, Cache-DiT, and TeaCache are not supported.
- No AMD GPU, Ascend NPU, or Intel GPU support is claimed.
- TAE (tiny autoencoder) fast decoding is not yet integrated; standard VAE decode is used.
- Reference images (5-view surround) are not yet integrated.

## References

- Checkpoint: <https://huggingface.co/acvlab/ABot-World-0-5B-LF>
- Official implementation: <https://github.com/amap-cvlab/ABot-World>
- Offline example: [`examples/offline_inference/diffusion/abot_world.py`](../../examples/offline_inference/diffusion/abot_world.py)
- Realtime example: [`examples/offline_inference/diffusion/abot_world_realtime.py`](../../examples/offline_inference/diffusion/abot_world_realtime.py)
- Realtime design: [`docs/design/feature/realtime_ar_diffusion.md`](../../docs/design/feature/realtime_ar_diffusion.md)
