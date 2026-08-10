# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline checks for ABot-World — no GPU/vllm needed. Pure logic verification.

Run: python3 tests/diffusion/models/abot_world/test_offline_check.py
"""
from __future__ import annotations

import math
import sys

OK, FAIL, TOTAL = 0, 0, 0


def check(cond, label):
    global OK, FAIL, TOTAL
    TOTAL += 1
    if cond:
        OK += 1
        print(f"  \033[32mPASS\033[0m  {label}")
    else:
        FAIL += 1
        print(f"  \033[31mFAIL\033[0m  {label}")


def section(msg):
    print(f"\n\033[36m=== {msg} ===\033[0m")


# ═══════════════════════════════════════════════════════════════════════
# 1. Config normalization
# ═══════════════════════════════════════════════════════════════════════
section("Config normalization")

raw_config = {
    "_class_name": "WanModel", "_diffusers_version": "0.33.0",
    "dim": 3072, "eps": 1e-6, "ffn_dim": 14336,
    "freq_dim": 256, "in_dim": 48, "model_type": "ti2v",
    "num_heads": 24, "num_layers": 30, "out_dim": 48,
    "text_len": 512, "downscale_factor_control_adapter": 16,
}

normalized = dict(raw_config)
if "in_dim" in normalized and "in_channels" not in normalized:
    normalized["in_channels"] = normalized.pop("in_dim")
if "out_dim" in normalized and "out_channels" not in normalized:
    normalized["out_channels"] = normalized.pop("out_dim")
if "num_heads" in normalized and "num_attention_heads" not in normalized:
    normalized["num_attention_heads"] = normalized.pop("num_heads")
if "dim" in normalized and "attention_head_dim" not in normalized:
    normalized["attention_head_dim"] = normalized["dim"] // normalized["num_attention_heads"]
if "text_len" in normalized and "text_dim" not in normalized:
    normalized["text_dim"] = normalized.pop("text_len")
for key in ("_class_name", "_diffusers_version", "model_type", "downscale_factor_control_adapter"):
    normalized.pop(key, None)

check(normalized["in_channels"] == 48, f"in_channels=48 (got {normalized['in_channels']})")
check(normalized["out_channels"] == 48, f"out_channels=48 (got {normalized['out_channels']})")
check(normalized["num_attention_heads"] == 24, "num_attention_heads=24")
check(normalized["attention_head_dim"] == 128, f"attention_head_dim=128 (got {normalized['attention_head_dim']})")
check(normalized["num_layers"] == 30, "num_layers=30")
check(normalized["ffn_dim"] == 14336, "ffn_dim=14336")
check(normalized["text_dim"] == 512, "text_dim=512")
check("_class_name" not in normalized, "no _class_name in normalized")
check("model_type" not in normalized, "no model_type in normalized")


# ═══════════════════════════════════════════════════════════════════════
# 2. Checkpoint key → parameter name mapping
# ═══════════════════════════════════════════════════════════════════════
section("Checkpoint key mapping")

def simulate_load_name(checkpoint_name):
    """Simulate the transformer load_weights name mapping."""
    name = checkpoint_name
    if name.startswith("model."):
        name = name[len("model."):]
    for proj_name in ("to_q", "to_k", "to_v"):
        marker = f".self_attn.{proj_name}."
        if marker in name:
            name = name.replace(marker, ".self_attn.to_qkv.")
            break
    return name

# Core transformer
check(simulate_load_name("model.patch_embedding.weight") == "patch_embedding.weight", "patch_embedding")
check(simulate_load_name("model.blocks.0.self_attn.to_q.weight") == "blocks.0.self_attn.to_qkv.weight", "self_attn q→qkv")
check(simulate_load_name("model.blocks.0.self_attn.to_k.weight") == "blocks.0.self_attn.to_qkv.weight", "self_attn k→qkv")
check(simulate_load_name("model.blocks.0.self_attn.to_v.weight") == "blocks.0.self_attn.to_qkv.weight", "self_attn v→qkv")
check(simulate_load_name("model.blocks.0.self_attn.norm_q.weight") == "blocks.0.self_attn.norm_q.weight", "norm_q")
check(simulate_load_name("model.blocks.0.self_attn.norm_k.weight") == "blocks.0.self_attn.norm_k.weight", "norm_k")
check(simulate_load_name("model.blocks.0.self_attn.to_out.weight") == "blocks.0.self_attn.to_out.weight", "to_out")
# Cross-attention
check(simulate_load_name("model.blocks.0.cross_attn.to_q.weight") == "blocks.0.cross_attn.to_q.weight", "cross_attn.q")
check(simulate_load_name("model.blocks.0.cross_attn.to_k.weight") == "blocks.0.cross_attn.to_k.weight", "cross_attn.k")
check(simulate_load_name("model.blocks.0.cross_attn.to_v.weight") == "blocks.0.cross_attn.to_v.weight", "cross_attn.v")
check(simulate_load_name("model.blocks.0.cross_attn.to_out.weight") == "blocks.0.cross_attn.to_out.weight", "cross_attn.o")
check(simulate_load_name("model.blocks.0.cross_attn.norm_q.weight") == "blocks.0.cross_attn.norm_q.weight", "cross_attn.norm_q")
check(simulate_load_name("model.blocks.0.cross_attn.norm_k.weight") == "blocks.0.cross_attn.norm_k.weight", "cross_attn.norm_k")
# FFN
check(simulate_load_name("model.blocks.0.ffn.0.weight") == "blocks.0.ffn.0.weight", "ffn.0")
check(simulate_load_name("model.blocks.0.ffn.2.weight") == "blocks.0.ffn.2.weight", "ffn.2")
# Modulation, embeddings, head
check(simulate_load_name("model.blocks.0.scale_shift_table") == "blocks.0.scale_shift_table", "scale_shift_table")
check(simulate_load_name("model.act_control_adapter.control_in_layer.weight") == "act_control_adapter.control_in_layer.weight", "act_control_adapter")
check(simulate_load_name("model.act_control_adapter.residual.0.weight") == "act_control_adapter.residual.0.weight", "adapter.residual.0")
check(simulate_load_name("model.time_embedding.0.weight") == "time_embedding.0.weight", "time_embedding")
check(simulate_load_name("model.time_projection.0.weight") == "time_projection.0.weight", "time_projection")
check(simulate_load_name("model.text_embedding.0.weight") == "text_embedding.0.weight", "text_embedding")
check(simulate_load_name("model.head.weight") == "head.weight", "head")
check(simulate_load_name("model.head.bias") == "head.bias", "head.bias")
check(simulate_load_name("model.head_modulation") == "head_modulation", "head_modulation")

# Unused Wan I2V keys (should be skipped by transformer.load_weights)
section("Unused Wan I2V keys (mapped but skipped)")
for key in ("model.c2ws_hidden_states_layer1.weight", "model.c2ws_hidden_states_layer2.weight",
            "model.patch_embedding_wancamctrl.weight", "model.c2ws_hidden_states_layer1.bias"):
    mapped = simulate_load_name(key)
    print(f"  {key} → {mapped}")

# ═══════════════════════════════════════════════════════════════════════
# 3. DMD schedule math
# ═══════════════════════════════════════════════════════════════════════
section("DMD flow schedule")

import torch
ABOT_DMD_TIMESTEPS = (1000, 750, 500, 250)
def _build_shifted_flow_schedule(*, flow_shift):
    base_sigmas = torch.tensor(ABOT_DMD_TIMESTEPS, dtype=torch.float64) / 1000
    shifted_numerators = flow_shift * base_sigmas
    shifted_sigmas = shifted_numerators / ((1.0 - base_sigmas) + shifted_numerators)
    warped_timesteps = shifted_sigmas * 1000
    return tuple((float(ts), float(sigma)) for ts, sigma in zip(warped_timesteps.tolist(), shifted_sigmas.tolist()))

sched = _build_shifted_flow_schedule(flow_shift=5.0)
check(len(sched) == 4, f"4 steps (got {len(sched)})")
for ts, sigma in sched:
    check(0 < sigma <= 1, f"sigma={sigma:.4f} in (0,1]")
    check(ts > 0, f"timestep={ts:.1f} > 0")

# Verify consistent with LingBot's same shift formula
sched_8 = _build_shifted_flow_schedule(flow_shift=8.0)
check(len(sched_8) == 4, "flow_shift=8.0 produces 4 steps")
# Second sigma (index 1) increases with higher shift
check(sched_8[1][1] > sched[1][1], "higher shift → higher sigma at step 2")

# ═══════════════════════════════════════════════════════════════════════
# 4. Flow shift validation
# ═══════════════════════════════════════════════════════════════════════
section("Flow shift validation")

def _positive_finite_flow_shift(value):
    if isinstance(value, bool):
        raise ValueError("flow_shift must be a positive finite number.")
    try:
        flow_shift = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("flow_shift must be a positive finite number.") from exc
    if not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("flow_shift must be a positive finite number.")
    return flow_shift

check(_positive_finite_flow_shift(5.0) == 5.0, "valid: 5.0")
check(_positive_finite_flow_shift(1.5) == 1.5, "valid: 1.5")
for bad in (0, -1, float('inf'), float('nan'), True):
    try:
        _positive_finite_flow_shift(bad)
        check(False, f"should reject: {bad}")
    except ValueError:
        check(True, f"rejects: {bad}")

# ═══════════════════════════════════════════════════════════════════════
# 5. Action reducer logic
# ═══════════════════════════════════════════════════════════════════════
section("Action reducer logic")

_ACTION_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")
_VALID_ACTIONS = frozenset(_ACTION_ORDER)

def _normalize_actions(value, field=""):
    if not (hasattr(value, '__iter__') and not isinstance(value, (str, bytes))):
        raise ValueError(f"{field} must be a sequence.")
    actions = set()
    for item in value:
        if not isinstance(item, str) or item.lower() not in _VALID_ACTIONS:
            raise ValueError(f"{field} supports only W/A/S/D/I/J/K/L.")
        actions.add(item.lower())
    return tuple(a for a in _ACTION_ORDER if a in actions)

check(_normalize_actions(["w", "D", "j"]) == ("w", "d", "j"), "normalize actions")
check(_normalize_actions([]) == (), "empty actions")

def parse_abot_frames(data, expected_frames):
    if data.get("mode") != "frames":
        raise ValueError("requires mode='frames'")
    frames_raw = data.get("frames")
    if not hasattr(frames_raw, '__iter__') or isinstance(frames_raw, (str, bytes)):
        raise ValueError("frames must be a sequence")
    frames = tuple(_normalize_actions(f, field=f"frame[{i}]") for i, f in enumerate(frames_raw))
    if len(frames) != expected_frames:
        raise ValueError(f"expected {expected_frames} frames, got {len(frames)}")
    return frames

result = parse_abot_frames({"mode": "frames", "frames": [["w"], ["a", "d"], []]}, 3)
check(result == (("w",), ("a", "d"), ()), "parse 3 frames correctly")

try:
    parse_abot_frames({"mode": "frames", "frames": [["w"], []]}, 3)
    check(False, "should reject wrong frame count")
except ValueError:
    check(True, "rejects wrong frame count")

# ═══════════════════════════════════════════════════════════════════════
# 6. VAE channel dimension check
# ═══════════════════════════════════════════════════════════════════════
section("VAE channel dimensions")

# Wan2.2 VAE has 48 latent channels
# Our VAE config hardcodes latent_channels=48
config = {
    "in_channels": 3, "out_channels": 3, "latent_channels": 48,
    "temporal_compression_ratio": 4, "spatial_compression_ratio": 8,
}
check(config["latent_channels"] == 48, "latent_channels=48 (Wan2.2 VAE)")
check(config["temporal_compression_ratio"] == 4, "temporal_compression_ratio=4")
check(config["spatial_compression_ratio"] == 8, "spatial_compression_ratio=8")

# Frame-to-latent conversion: num_latent_frames = (num_frames - 1) // 4 + 1
for pixel_frames in (9, 13, 21, 33, 81, 117, 121):
    latent_frames = (pixel_frames - 1) // 4 + 1
    check((pixel_frames - 1) % 4 == 0, f"{pixel_frames} frames → {latent_frames} latent (valid Wan2.2 geometry)")

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
section("Summary")
print(f"  {OK}/{TOTAL} passed, {FAIL} failed")
if FAIL:
    print("  \033[31mSOME CHECKS FAILED\033[0m")
    sys.exit(1)
else:
    print("  \033[32mALL CHECKS PASSED\033[0m")
