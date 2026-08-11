# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for ABot-World configuration and helpers."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.diffusion.models.abot_world.pipeline import (
    ABOT_DMD_TIMESTEPS,
    _build_shifted_flow_schedule,
    _convert_wan_umt5_encoder_state_dict,
    _positive_finite_flow_shift,
    _resolve_local_model_path,
    _validate_local_model_files,
)
from vllm_omni.diffusion.models.abot_world.transformer import (
    ABotWorldCausalTransformer3DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_shifted_flow_schedule_is_finite_and_monotonic() -> None:
    schedule = _build_shifted_flow_schedule(flow_shift=5.0)

    assert len(schedule) == len(ABOT_DMD_TIMESTEPS) == 4
    assert all(math.isfinite(timestep) and math.isfinite(sigma) for timestep, sigma in schedule)
    assert all(0 < sigma <= 1 for _, sigma in schedule)
    assert [sigma for _, sigma in schedule] == sorted(
        (sigma for _, sigma in schedule), reverse=True
    )


@pytest.mark.parametrize("value", [0, -1, True, math.inf, math.nan, "invalid"])
def test_flow_shift_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _positive_finite_flow_shift(value)


def test_local_model_path_accepts_existing_directory(tmp_path) -> None:
    assert _resolve_local_model_path(str(tmp_path)) == str(tmp_path.resolve())


def test_incomplete_local_checkpoint_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="diffusion_pytorch_model.safetensors"):
        _validate_local_model_files(str(tmp_path))


def test_umt5_conversion_maps_gated_gelu_branches() -> None:
    source = {
        "token_embedding.weight": torch.zeros(2, 2),
        "norm.weight": torch.zeros(2),
    }
    suffixes = (
        "norm1.weight",
        "attn.q.weight",
        "attn.k.weight",
        "attn.v.weight",
        "attn.o.weight",
        "pos_embedding.embedding.weight",
        "norm2.weight",
        "ffn.gate.0.weight",
        "ffn.fc1.weight",
        "ffn.fc2.weight",
    )
    for value, suffix in enumerate(suffixes):
        source[f"blocks.0.{suffix}"] = torch.full((1,), value)

    converted = _convert_wan_umt5_encoder_state_dict(source, num_layers=1)

    assert converted[
        "encoder.block.0.layer.1.DenseReluDense.wi_0.weight"
    ].item() == suffixes.index("ffn.gate.0.weight")
    assert converted[
        "encoder.block.0.layer.1.DenseReluDense.wi_1.weight"
    ].item() == suffixes.index("ffn.fc1.weight")


def test_from_config_uses_wan_text_width_not_text_length() -> None:
    model = ABotWorldCausalTransformer3DModel.from_config(
        {
            "patch_size": [1, 2, 2],
            "num_heads": 1,
            "dim": 8,
            "ffn_dim": 16,
            "num_layers": 1,
            "in_dim": 4,
            "out_dim": 4,
            "text_len": 512,
            "downscale_factor_control_adapter": 2,
        }
    )

    assert model.config.text_dim == 4096
    assert model.config.downscale_factor_control_adapter == 2
    names = dict(model.named_parameters())
    assert "blocks.0.modulation" in names
    assert "blocks.0.self_attn.qkv.weight" in names
    assert "blocks.0.self_attn.o.weight" in names
    assert "blocks.0.cross_attn.q.weight" in names
    assert "blocks.0.norm3.weight" in names
    assert "blocks.0.norm3.bias" in names
    assert "act_control_adapter.conv.weight" in names
    assert "act_control_adapter.residual_blocks.0.conv1.weight" in names
    assert "head.modulation" in names
    assert "head.head.weight" in names
