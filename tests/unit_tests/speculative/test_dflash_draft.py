# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the DFlash draft model and its helpers."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    Qwen3DFlashDraftModel,
    _merge_multiaxis_rotary_embeddings,
    _prepend_text_position_ids,
    apply_rotary_pos_emb,
    build_target_layer_ids,
    extract_context_feature,
)


def test_build_target_layer_ids_spread_and_count():
    # single draft layer -> middle of the target
    assert build_target_layer_ids(36, 1) == [18]
    # multiple draft layers -> monotonic, in-bounds spread
    ids = build_target_layer_ids(36, 5)
    assert len(ids) == 5
    assert ids == sorted(ids)
    assert all(0 <= i < 36 for i in ids)


def test_extract_context_feature_uses_offset_one():
    # hidden_states[0] is the embedding output; layer i's output is at index i+1.
    hs = [torch.full((1, 2, 3), float(i)) for i in range(6)]
    out = extract_context_feature(hs, [1, 3])
    assert out.shape == (1, 2, 6)
    # first 3 features come from hidden_states[2], next 3 from hidden_states[4]
    assert torch.allclose(out[..., :3], torch.full((1, 2, 3), 2.0))
    assert torch.allclose(out[..., 3:], torch.full((1, 2, 3), 4.0))


def test_prepend_text_position_ids_adds_causal_mask_axis() -> None:
    """Qwen-VL decode positions include text plus three MRoPE axes."""
    multimodal = torch.tensor(
        [
            [[7, 8, 9, 10]],
            [[11, 12, 13, 14]],
            [[15, 16, 17, 18]],
        ]
    )
    attention_mask = torch.tensor([[0, 1, 1, 1]])

    position_ids = _prepend_text_position_ids(multimodal, attention_mask)

    assert position_ids.shape == (4, 1, 4)
    torch.testing.assert_close(position_ids[0], torch.tensor([[0, 0, 1, 2]]))
    torch.testing.assert_close(position_ids[1:], multimodal)


def _draft_cfg(bs=4):
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1000000,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = 8
    cfg.block_size = bs
    cfg.dflash_config = {"mask_token_id": 63, "target_layer_ids": [1, 3, 5]}
    cfg._attn_implementation = "sdpa"
    return cfg


def _rope_relerr(inv_freq):
    # Reference fp32 default-rope frequencies for head_dim=8, theta=1e6.
    ref = 1.0 / (1000000 ** (torch.arange(0, 8, 2).float() / 8))
    return ((inv_freq.float() - ref).abs() / ref).max().item()


def test_rope_inv_freq_stays_fp32_after_bf16_cast():
    """``model.to(bf16)`` must not round the RoPE frequencies.

    The serving runtime keeps an fp32 RoPE cache; if the draft trained with a
    bf16-rounded ``inv_freq`` the train/inference RoPE would diverge (worse with
    longer context) and erode acceptance. The draft pins ``inv_freq`` to fp32.
    """
    draft = Qwen3DFlashDraftModel(_draft_cfg()).to(torch.bfloat16)
    inv_freq = draft.rotary_emb.inv_freq
    assert inv_freq.dtype == torch.float32
    # Recomputed fresh, so the values are exact fp32 (not a bf16 round-trip).
    assert _rope_relerr(inv_freq) < 1e-6
    # original_inv_freq (used by dynamic-rope resets) is pinned too.
    assert draft.rotary_emb.original_inv_freq.dtype == torch.float32
    # The rest of the model is still bf16 (the pin is rope-only).
    assert next(draft.layers[0].parameters()).dtype == torch.bfloat16


def test_rope_inv_freq_fp32_survives_chained_casts():
    draft = Qwen3DFlashDraftModel(_draft_cfg()).to(torch.float16).to(torch.bfloat16)
    assert draft.rotary_emb.inv_freq.dtype == torch.float32
    assert _rope_relerr(draft.rotary_emb.inv_freq) < 1e-6


def test_draft_forward_output_shape():
    H, n_layers_target, tli, bs = 32, 8, [1, 3, 5], 4
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=H,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = n_layers_target
    cfg.block_size = bs
    cfg.dflash_config = {"mask_token_id": 63, "target_layer_ids": tli}
    cfg._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(cfg)
    assert draft.target_layer_ids == tli
    assert draft.fc.in_features == len(tli) * H

    B, S, N = 2, 10, 3
    Q = N * bs
    noise = torch.randn(B, Q, H)
    target_hidden = torch.randn(B, S, len(tli) * H)
    position_ids = torch.arange(S + Q).unsqueeze(0).expand(B, -1)
    out = draft(position_ids=position_ids, attention_mask=None, noise_embedding=noise, target_hidden=target_hidden)
    assert out.shape == (B, Q, H)
    assert torch.isfinite(out).all()


def test_plain_dflash_sampling_keeps_parallel_suffix_logits() -> None:
    """A non-Domino checkpoint samples all suffix positions from one LM-head result."""
    torch.manual_seed(13)
    draft = Qwen3DFlashDraftModel(_draft_cfg()).eval()
    target = nn.Module()
    target.lm_head = nn.Linear(32, 64, bias=False)
    draft_hidden = torch.randn(1, 4, 32)
    block_ids = torch.tensor([[5, 63, 63, 63]])

    with torch.inference_mode():
        expected = target.lm_head(draft_hidden[:, 1:, :]).argmax(dim=-1)
        actual = draft._sample_draft_tokens(target, draft_hidden, block_ids)

    torch.testing.assert_close(actual, expected)


def _routing_cfg() -> Qwen3Config:
    config = _draft_cfg()
    config.dflash_config = {
        "mask_token_id": 63,
        "target_layer_ids": [1, 5],
        "layer_routing_enabled": True,
    }
    return config


def test_zero_layer_routing_gates_preserve_plain_dflash_output() -> None:
    """A routing warm start must be numerically identical before training."""
    torch.manual_seed(23)
    plain_config = deepcopy(_routing_cfg())
    plain_config.dflash_config["layer_routing_enabled"] = False
    plain = Qwen3DFlashDraftModel(plain_config).eval()
    routed = Qwen3DFlashDraftModel(_routing_cfg()).eval()
    missing, unexpected = routed.load_state_dict(plain.state_dict(), strict=False)
    assert missing == ["layer_route_gates"]
    assert unexpected == []

    batch_size, context_length, query_length = 2, 7, 8
    target_hidden = torch.randn(batch_size, context_length, 64)
    noise = torch.randn(batch_size, query_length, 32)
    position_ids = torch.arange(context_length + query_length).unsqueeze(0).expand(batch_size, -1)

    with torch.inference_mode():
        plain_output = plain(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )
        routed_output = routed(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )

    torch.testing.assert_close(routed_output, plain_output, rtol=0, atol=0)


def test_layer_routing_stage_trains_only_gates() -> None:
    """The diagnostic stage changes only the three-way context mixture."""
    model = Qwen3DFlashDraftModel(_routing_cfg())
    model.set_layer_routing_training_stage("layer_routing")
    target_hidden = torch.randn(2, 7, 64)
    noise = torch.randn(2, 8, 32)
    position_ids = torch.arange(15).unsqueeze(0).expand(2, -1)

    output = model(position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden)
    output.square().mean().backward()

    assert model.layer_route_gates.requires_grad
    assert model.layer_route_gates.grad is not None
    assert torch.isfinite(model.layer_route_gates.grad).all()
    assert torch.count_nonzero(model.layer_route_gates.grad) > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name != "layer_route_gates")


def test_layer_routing_requires_one_target_layer_per_draft_layer() -> None:
    config = _draft_cfg()
    config.dflash_config["layer_routing_enabled"] = True

    with pytest.raises(ValueError, match="one captured target layer per draft layer"):
        Qwen3DFlashDraftModel(config)


def _spatial_rope_cfg() -> Qwen3Config:
    config = _draft_cfg()
    config.dflash_config = {
        "mask_token_id": 63,
        "target_layer_ids": [1, 3, 5],
        "spatial_rope_enabled": True,
        "spatial_rope_sections": [1, 1, 2],
    }
    return config


def _hard_spatial_rope_cfg() -> Qwen3Config:
    config = _spatial_rope_cfg()
    config.dflash_config["spatial_rope_mode"] = "replace"
    return config


def test_equal_multimodal_axes_match_text_rotary_embeddings() -> None:
    model = Qwen3DFlashDraftModel(_spatial_rope_cfg()).eval()
    hidden_states = torch.randn(2, 4, 32)
    position_ids = torch.arange(9).unsqueeze(0).expand(2, -1)

    expected = model.rotary_emb(hidden_states, position_ids)
    actual = _merge_multiaxis_rotary_embeddings(
        model.rotary_emb,
        hidden_states,
        position_ids.unsqueeze(0).expand(3, -1, -1),
        model.spatial_rope_sections,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_multimodal_rotary_sections_match_qwen_vl_reference() -> None:
    model = Qwen3DFlashDraftModel(_spatial_rope_cfg()).eval()
    hidden_states = torch.randn(2, 4, 32)
    position_ids = torch.randint(0, 16, (3, 2, 9))
    merged_cos, merged_sin = _merge_multiaxis_rotary_embeddings(
        model.rotary_emb,
        hidden_states,
        position_ids,
        model.spatial_rope_sections,
    )
    flat_cos, flat_sin = model.rotary_emb(hidden_states, position_ids.flatten(0, 1))
    axis_cos = flat_cos.reshape(3, 2, 9, 8)
    axis_sin = flat_sin.reshape(3, 2, 9, 8)
    query = torch.randn(2, 4, 9, 8)
    key = torch.randn(2, 2, 9, 8)

    actual = apply_rotary_pos_emb(query, key, merged_cos, merged_sin)
    expected = apply_multimodal_rotary_pos_emb(
        query,
        key,
        axis_cos,
        axis_sin,
        list(model.spatial_rope_sections),
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_zero_spatial_rope_gates_preserve_plain_dflash_output() -> None:
    torch.manual_seed(29)
    plain = Qwen3DFlashDraftModel(_draft_cfg()).eval()
    spatial = Qwen3DFlashDraftModel(_spatial_rope_cfg()).eval()
    missing, unexpected = spatial.load_state_dict(plain.state_dict(), strict=False)
    assert missing == ["spatial_rope_gates"]
    assert unexpected == []
    target_hidden = torch.randn(2, 7, 96)
    noise = torch.randn(2, 8, 32)
    position_ids = torch.arange(15).unsqueeze(0).expand(2, -1)
    multimodal_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()
    multimodal_position_ids[1, :, 2:5] = torch.tensor([3, 4, 3])
    multimodal_position_ids[2, :, 2:5] = torch.tensor([7, 7, 8])

    with torch.inference_mode():
        expected = plain(position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden)
        actual = spatial(
            position_ids=position_ids,
            multimodal_position_ids=multimodal_position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_spatial_rope_stage_trains_only_gates() -> None:
    model = Qwen3DFlashDraftModel(_spatial_rope_cfg())
    model.set_spatial_rope_training_stage("spatial_rope")
    position_ids = torch.arange(15).unsqueeze(0).expand(2, -1)
    multimodal_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()
    multimodal_position_ids[1, :, 1:6] += 2
    multimodal_position_ids[2, :, 1:6] += 5

    output = model(
        position_ids=position_ids,
        multimodal_position_ids=multimodal_position_ids,
        noise_embedding=torch.randn(2, 8, 32),
        target_hidden=torch.randn(2, 7, 96),
    )
    output.square().mean().backward()

    assert model.spatial_rope_gates.grad is not None
    assert torch.isfinite(model.spatial_rope_gates.grad).all()
    assert torch.count_nonzero(model.spatial_rope_gates.grad) > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name != "spatial_rope_gates")


def test_hard_spatial_rope_uses_no_gate_and_changes_spatial_output() -> None:
    torch.manual_seed(31)
    plain = Qwen3DFlashDraftModel(_draft_cfg()).eval()
    spatial = Qwen3DFlashDraftModel(_hard_spatial_rope_cfg()).eval()
    missing, unexpected = spatial.load_state_dict(plain.state_dict(), strict=False)
    assert missing == []
    assert unexpected == []
    assert spatial.spatial_rope_gates is None
    assert "spatial_rope_gates" not in spatial.state_dict()
    target_hidden = torch.randn(2, 7, 96)
    noise = torch.randn(2, 8, 32)
    position_ids = torch.arange(15).unsqueeze(0).expand(2, -1)
    multimodal_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()
    multimodal_position_ids[1, :, 1:7] += 3
    multimodal_position_ids[2, :, 1:7] += 7

    with torch.inference_mode():
        expected = plain(position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden)
        actual = spatial(
            position_ids=position_ids,
            multimodal_position_ids=multimodal_position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )

    assert not torch.equal(actual, expected)
    assert torch.isfinite(actual).all()


def test_hard_spatial_rope_joint_stage_trains_backbone() -> None:
    model = Qwen3DFlashDraftModel(_hard_spatial_rope_cfg())
    with pytest.raises(ValueError, match="requires training_stage='joint'"):
        model.set_spatial_rope_training_stage("spatial_rope")
    model.set_spatial_rope_training_stage("joint")
    position_ids = torch.arange(15).unsqueeze(0).expand(2, -1)
    multimodal_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()
    multimodal_position_ids[1, :, 1:6] += 2
    multimodal_position_ids[2, :, 1:6] += 5

    output = model(
        position_ids=position_ids,
        multimodal_position_ids=multimodal_position_ids,
        noise_embedding=torch.randn(2, 8, 32),
        target_hidden=torch.randn(2, 7, 96),
    )
    output.square().mean().backward()

    gradient = model.layers[0].self_attn.q_proj.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_draft_sdpa_matches_eager_in_fp32():
    """The inference SDPA path must preserve the eager attention result."""
    torch.manual_seed(7)
    eager_cfg = _draft_cfg()
    eager_cfg._attn_implementation = "eager"
    sdpa_cfg = deepcopy(eager_cfg)
    sdpa_cfg._attn_implementation = "sdpa"

    eager_draft = Qwen3DFlashDraftModel(eager_cfg).eval()
    sdpa_draft = Qwen3DFlashDraftModel(sdpa_cfg).eval()
    sdpa_draft.load_state_dict(eager_draft.state_dict())

    batch_size, context_length, num_blocks = 2, 7, 2
    query_length = num_blocks * eager_cfg.block_size
    noise = torch.randn(batch_size, query_length, eager_cfg.hidden_size)
    target_hidden = torch.randn(
        batch_size,
        context_length,
        len(eager_cfg.dflash_config["target_layer_ids"]) * eager_cfg.hidden_size,
    )
    position_ids = torch.arange(context_length + query_length).unsqueeze(0).expand(batch_size, -1)

    with torch.inference_mode():
        eager_output = eager_draft(
            position_ids=position_ids,
            attention_mask=None,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )
        sdpa_output = sdpa_draft(
            position_ids=position_ids,
            attention_mask=None,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )

    torch.testing.assert_close(sdpa_output, eager_output, rtol=1e-5, atol=1e-6)
