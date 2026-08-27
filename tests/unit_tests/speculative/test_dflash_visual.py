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

"""CPU tests for DFlash's per-layer cached visual adaptor."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule
from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    Qwen3DFlashDraftModel,
    _ablate_image_token_hidden_states,
    _bounded_acceptance_counts,
)
from nemo_automodel.recipes.llm import train_dflash
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe


def _config(*, visual: bool, visual_adapter_type: str = "cross_attention") -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config.num_target_layers = 8
    config.block_size = 4
    config.dflash_config = {
        "mask_token_id": 63,
        "target_layer_ids": [1, 5],
    }
    if visual:
        config.dflash_config.update(
            {
                "visual_num_query_tokens": 2,
                "visual_adapter_type": visual_adapter_type,
                "visual_adapter_dim": 16,
                "visual_num_attention_heads": 4,
                "visual_gate_init": 1.0e-3,
                "visual_image_token_id": 62,
            }
        )
    config._attn_implementation = "sdpa"
    return config


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    target_hidden = torch.randn(2, 6, 64)
    noise_embedding = torch.randn(2, 4, 32)
    position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
    image_mask = torch.tensor(
        [
            [False, True, True, False, False, False],
            [False, False, True, True, False, False],
        ]
    )
    return target_hidden, noise_embedding, position_ids, image_mask


def test_image_token_hidden_state_ablation_preserves_non_image_rows() -> None:
    target_hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    image_mask = torch.tensor([[False, True, True, True, False]])

    zeroed = _ablate_image_token_hidden_states(target_hidden, image_mask, "zero")
    shuffled = _ablate_image_token_hidden_states(target_hidden, image_mask, "shuffle")

    torch.testing.assert_close(zeroed[:, [0, 4]], target_hidden[:, [0, 4]])
    assert torch.count_nonzero(zeroed[:, 1:4]) == 0
    torch.testing.assert_close(shuffled[:, [0, 4]], target_hidden[:, [0, 4]])
    torch.testing.assert_close(shuffled[:, 1:4], target_hidden[:, 1:4].flip(1))
    torch.testing.assert_close(target_hidden, torch.arange(20, dtype=torch.float32).reshape(1, 5, 4))


def test_image_token_hidden_state_ablation_keep_returns_original_tensor() -> None:
    target_hidden = torch.randn(1, 3, 4)
    image_mask = torch.tensor([[False, True, False]])

    assert _ablate_image_token_hidden_states(target_hidden, image_mask, "keep") is target_hidden


def test_acceptance_counts_exclude_draft_positions_past_requested_length() -> None:
    assert _bounded_acceptance_counts(start=10, max_length=20, block_size=8, acceptance_length=4) == (7, 4)
    assert _bounded_acceptance_counts(start=17, max_length=20, block_size=8, acceptance_length=7) == (2, 2)
    assert _bounded_acceptance_counts(start=19, max_length=20, block_size=8, acceptance_length=7) == (0, 0)


def test_gate_zero_matches_pre_visual_checkpoint() -> None:
    torch.manual_seed(11)
    baseline = Qwen3DFlashDraftModel(_config(visual=False)).eval()
    visual = Qwen3DFlashDraftModel(_config(visual=True)).eval()
    missing, unexpected = visual.load_state_dict(copy.deepcopy(baseline.state_dict()), strict=False)
    assert missing
    assert all(".visual_fusion." in name for name in missing)
    assert not unexpected
    with torch.no_grad():
        for layer in visual.layers:
            layer.visual_fusion.gate.zero_()

    target_hidden, noise_embedding, position_ids, image_mask = _inputs()
    expected = baseline(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
    )
    actual = visual(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        image_mask=image_mask,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pooled_mlp_gate_zero_matches_pre_visual_checkpoint() -> None:
    torch.manual_seed(11)
    baseline = Qwen3DFlashDraftModel(_config(visual=False)).eval()
    visual = Qwen3DFlashDraftModel(_config(visual=True, visual_adapter_type="pooled_mlp")).eval()
    missing, unexpected = visual.load_state_dict(copy.deepcopy(baseline.state_dict()), strict=False)
    assert missing
    assert all(".visual_fusion." in name for name in missing)
    assert not unexpected
    assert all(layer.visual_fusion.gate.item() == 0.0 for layer in visual.layers)

    target_hidden, noise_embedding, position_ids, image_mask = _inputs()
    expected = baseline(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
    )
    actual = visual(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        image_mask=image_mask,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pooled_mlp_context_is_masked_mean_followed_by_target_mlp() -> None:
    model = Qwen3DFlashDraftModel(_config(visual=True, visual_adapter_type="pooled_mlp")).eval()
    target_hidden, _, _, image_mask = _inputs()
    layer = model.layers[0].visual_fusion

    context = layer.build_context(target_hidden[:, :, :32], image_mask)
    mask = image_mask.unsqueeze(-1).to(target_hidden.dtype)
    expected_pool = (layer.target_norm(target_hidden[:, :, :32]) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    expected = layer.target_mlp(expected_pool).unsqueeze(1)

    assert context.shape == (2, 1, 16)
    torch.testing.assert_close(context, expected)


def test_pooled_mlp_nonzero_gate_uses_visual_context() -> None:
    model = Qwen3DFlashDraftModel(_config(visual=True, visual_adapter_type="pooled_mlp")).eval()
    target_hidden, noise_embedding, position_ids, image_mask = _inputs()
    with torch.no_grad():
        for layer in model.layers:
            layer.visual_fusion.gate.fill_(0.5)

    with_visual = model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        image_mask=image_mask,
    )
    without_visual = model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        image_mask=torch.zeros_like(image_mask),
    )

    assert not torch.equal(with_visual, without_visual)


def test_unknown_visual_adapter_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="visual_adapter_type"):
        Qwen3DFlashDraftModel(_config(visual=True, visual_adapter_type="unknown"))


def test_each_draft_layer_uses_only_its_paired_target_layer() -> None:
    model = Qwen3DFlashDraftModel(_config(visual=True)).eval()
    target_hidden, _, _, image_mask = _inputs()
    original = model.build_visual_context(target_hidden, image_mask)
    changed = target_hidden.clone()
    changed[:, :, :32] += 10.0
    perturbed = model.build_visual_context(changed, image_mask)

    assert not torch.equal(original[0], perturbed[0])
    torch.testing.assert_close(original[1], perturbed[1], rtol=0, atol=0)


def test_text_only_sample_has_finite_zero_visual_context() -> None:
    model = Qwen3DFlashDraftModel(_config(visual=True)).eval()
    target_hidden, _, _, _ = _inputs()
    image_mask = torch.zeros(target_hidden.shape[:2], dtype=torch.bool)
    contexts = model.build_visual_context(target_hidden, image_mask)

    assert len(contexts) == len(model.layers)
    for context in contexts:
        assert torch.isfinite(context).all()
        assert torch.count_nonzero(context) == 0


@pytest.mark.parametrize("visual_adapter_type", ["cross_attention", "pooled_mlp"])
def test_visual_adaptor_stage_freezes_backbone_and_backpropagates(visual_adapter_type: str) -> None:
    model = Qwen3DFlashDraftModel(_config(visual=True, visual_adapter_type=visual_adapter_type))
    model.set_training_stage("visual_adaptor")
    target_hidden, noise_embedding, position_ids, image_mask = _inputs()
    output = model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        image_mask=image_mask,
    )
    output.square().mean().backward()

    visual_parameters = model.visual_parameters()
    backbone_parameters = model.backbone_parameters()
    assert visual_parameters and all(parameter.requires_grad for parameter in visual_parameters)
    assert backbone_parameters and all(not parameter.requires_grad for parameter in backbone_parameters)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in visual_parameters)
    assert all(parameter.grad is None for parameter in backbone_parameters)


def test_joint_stage_uses_distinct_backbone_and_visual_parameter_sets() -> None:
    model = Qwen3DFlashDraftModel(_config(visual=True))
    model.set_training_stage("joint")
    visual_parameters = model.visual_parameters()
    backbone_parameters = model.backbone_parameters()

    assert all(parameter.requires_grad for parameter in (*visual_parameters, *backbone_parameters))
    assert {id(parameter) for parameter in visual_parameters}.isdisjoint(
        {id(parameter) for parameter in backbone_parameters}
    )


def test_visual_initialization_is_reproducible_from_training_seed() -> None:
    torch.manual_seed(42)
    first = Qwen3DFlashDraftModel(_config(visual=True))
    torch.manual_seed(42)
    second = Qwen3DFlashDraftModel(_config(visual=True))

    for first_parameter, second_parameter in zip(first.visual_parameters(), second.visual_parameters()):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_recipe_warm_start_accepts_only_new_visual_keys(monkeypatch) -> None:
    baseline = Qwen3DFlashDraftModel(_config(visual=False))
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(_config(visual=True))
    monkeypatch.setattr(
        train_dflash,
        "load_hf_safetensors_state_dict",
        lambda _path: copy.deepcopy(baseline.state_dict()),
    )

    recipe._load_draft_init("old-dflash")


def test_recipe_warm_start_accepts_new_layer_routing_gates(monkeypatch) -> None:
    baseline = Qwen3DFlashDraftModel(_config(visual=False))
    routing_config = _config(visual=False)
    routing_config.dflash_config["layer_routing_enabled"] = True
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(routing_config)
    monkeypatch.setattr(
        train_dflash,
        "load_hf_safetensors_state_dict",
        lambda _path: copy.deepcopy(baseline.state_dict()),
    )

    recipe._load_draft_init("old-dflash")

    torch.testing.assert_close(recipe.draft_model.layer_route_gates, torch.zeros(2))


def test_recipe_warm_start_accepts_new_spatial_rope_gates(monkeypatch) -> None:
    baseline = Qwen3DFlashDraftModel(_config(visual=False))
    spatial_config = _config(visual=False)
    spatial_config.dflash_config.update(
        {
            "spatial_rope_enabled": True,
            "spatial_rope_sections": [1, 1, 2],
        }
    )
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(spatial_config)
    monkeypatch.setattr(
        train_dflash,
        "load_hf_safetensors_state_dict",
        lambda _path: copy.deepcopy(baseline.state_dict()),
    )

    recipe._load_draft_init("old-dflash")

    torch.testing.assert_close(recipe.draft_model.spatial_rope_gates, torch.zeros(2))


def test_recipe_warm_start_hard_spatial_rope_has_no_new_parameters(monkeypatch) -> None:
    baseline = Qwen3DFlashDraftModel(_config(visual=False))
    spatial_config = _config(visual=False)
    spatial_config.dflash_config.update(
        {
            "spatial_rope_enabled": True,
            "spatial_rope_mode": "replace",
            "spatial_rope_sections": [1, 1, 2],
        }
    )
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(spatial_config)
    monkeypatch.setattr(
        train_dflash,
        "load_hf_safetensors_state_dict",
        lambda _path: copy.deepcopy(baseline.state_dict()),
    )

    recipe._load_draft_init("old-dflash")

    assert recipe.draft_model.spatial_rope_gates is None


def test_recipe_warm_start_restores_exact_base_draft(monkeypatch) -> None:
    expected = Qwen3DFlashDraftModel(_config(visual=False))
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(_config(visual=False))
    with torch.no_grad():
        for parameter in recipe.draft_model.parameters():
            parameter.zero_()
    monkeypatch.setattr(
        train_dflash,
        "load_hf_safetensors_state_dict",
        lambda _path: copy.deepcopy(expected.state_dict()),
    )

    recipe._load_draft_init("base-dflash")

    for actual_parameter, expected_parameter in zip(recipe.draft_model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)


def test_recipe_stamps_pooled_mlp_visual_adapter_type() -> None:
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.mask_token_id = 63
    recipe.spatial_rope_enabled = False
    recipe.spatial_rope_mode = "gated"
    recipe.spatial_rope_sections = []
    recipe.layer_routing_enabled = False
    recipe.visual_num_query_tokens = 1
    recipe.visual_image_token_id = 62

    config = recipe._build_dflash_config(
        {
            "visual_adapter_type": "pooled_mlp",
            "visual_adapter_dim": 128,
        },
        [1, 5],
    )

    assert config["visual_adapter_type"] == "pooled_mlp"
    assert config["visual_adapter_dim"] == 128


def test_recipe_warm_start_rejects_missing_backbone_weight(monkeypatch) -> None:
    baseline = Qwen3DFlashDraftModel(_config(visual=False))
    state_dict = copy.deepcopy(baseline.state_dict())
    state_dict.pop("fc.weight")
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(_config(visual=True))
    monkeypatch.setattr(train_dflash, "load_hf_safetensors_state_dict", lambda _path: state_dict)

    with pytest.raises(ValueError, match="fc.weight"):
        recipe._load_draft_init("incomplete-dflash")


def test_training_rejects_image_after_supervised_token() -> None:
    model = Qwen3DFlashDraftModel(_config(visual=True))
    trainer = DFlashTrainerModule(
        draft_model=model,
        target_lm_head=torch.nn.Linear(32, 64, bias=False),
        target_embed_tokens=torch.nn.Embedding(64, 32),
        mask_token_id=63,
        block_size=4,
        attention_backend="sdpa",
        num_anchors=2,
    )
    input_ids = torch.randint(0, 63, (1, 8))
    hidden_states = torch.randn(1, 8, 64)
    loss_mask = torch.tensor([[False, False, True, True, True, True, True, True]])
    image_mask = torch.tensor([[False, True, False, False, True, False, False, False]])

    with pytest.raises(ValueError, match="future visual information"):
        trainer(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            image_mask=image_mask,
        )


def test_spatial_rope_trainer_builds_block_positions_and_backpropagates() -> None:
    config = _config(visual=False)
    config.dflash_config.update(
        {
            "spatial_rope_enabled": True,
            "spatial_rope_sections": [1, 1, 2],
        }
    )
    model = Qwen3DFlashDraftModel(config)
    model.set_spatial_rope_training_stage("spatial_rope")
    trainer = DFlashTrainerModule(
        draft_model=model,
        target_lm_head=torch.nn.Linear(32, 64, bias=False),
        target_embed_tokens=torch.nn.Embedding(64, 32),
        mask_token_id=63,
        block_size=4,
        attention_backend="sdpa",
        num_anchors=2,
    )
    input_ids = torch.randint(0, 63, (1, 12))
    position_ids = torch.arange(12).unsqueeze(0)
    multimodal_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()
    multimodal_position_ids[1, :, 2:6] += 2
    multimodal_position_ids[2, :, 2:6] += 4

    metrics = trainer(
        input_ids=input_ids,
        hidden_states=torch.randn(1, 12, 64),
        loss_mask=torch.ones(1, 12, dtype=torch.bool),
        multimodal_position_ids=multimodal_position_ids,
    )
    metrics.loss.backward()

    assert torch.isfinite(metrics.loss)
    assert model.spatial_rope_gates.grad is not None
    assert torch.isfinite(model.spatial_rope_gates.grad).all()
