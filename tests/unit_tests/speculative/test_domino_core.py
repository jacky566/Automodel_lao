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

"""Tests for the Domino trainer module (causal head, dual-logit loss, curriculum)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.core import compute_accept_len
from nemo_automodel.components.speculative.dflash.domino_core import (
    DominoStepMetrics,
    DominoTrainerModule,
    get_lambda_base,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.recipes.llm import train_domino
from nemo_automodel.recipes.llm.train_domino import TrainDominoRecipe

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
EMB_DIM = 16
GRU_HIDDEN = 24


class _TinyDominoTarget(nn.Module):
    """Target-side embedding and LM head needed by draft token sampling."""

    def __init__(self, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def get_input_embeddings(self) -> nn.Module:
        """Return the target token embedding module."""
        return self.embed_tokens


def _draft_config(projector_type="domino", pure_prefix=1, shift_label=True, spatial_rope=False):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
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
    cfg.num_target_layers = NUM_TARGET_LAYERS
    cfg.block_size = BLOCK_SIZE
    dflash_config = {"mask_token_id": MASK_ID, "target_layer_ids": TARGET_LAYER_IDS}
    if projector_type is not None:
        dflash_config.update(
            {
                "projector_type": projector_type,
                "emb_dim": EMB_DIM,
                "gru_hidden_dim": GRU_HIDDEN,
                "pure_draft_prefix_len": pure_prefix,
                "shift_label": shift_label,
            }
        )
    if spatial_rope:
        dflash_config.update(
            {
                "spatial_rope_enabled": True,
                "spatial_rope_mode": "replace",
                "spatial_rope_sections": [1, 1, 2],
            }
        )
    cfg.dflash_config = dflash_config
    cfg._attn_implementation = "sdpa"
    return cfg


def _build_trainer(num_anchors=8, loss_decay_gamma=None, pure_prefix=1, shift_label=True):
    draft = Qwen3DFlashDraftModel(_draft_config(pure_prefix=pure_prefix, shift_label=shift_label))
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    return DominoTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        shift_label=shift_label,
    )


def _inputs(bsz=2, seq_len=24):
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB - 1, (bsz, seq_len))
    loss_mask = torch.ones(bsz, seq_len)
    hidden = torch.randn(bsz, seq_len, len(TARGET_LAYER_IDS) * HIDDEN)
    return input_ids, hidden, loss_mask


# --------------------------------------------------------------------------- #
# Draft head construction
# --------------------------------------------------------------------------- #


def test_draft_builds_domino_head():
    draft = Qwen3DFlashDraftModel(_draft_config())
    assert isinstance(draft.prefix_gru, torch.nn.GRU)
    assert draft.prefix_gru.input_size == HIDDEN
    assert draft.prefix_gru.hidden_size == GRU_HIDDEN
    # embed_proj projects [hidden | gru] -> emb_dim -> vocab.
    assert draft.embed_proj[0].in_features == HIDDEN + GRU_HIDDEN
    assert draft.embed_proj[-1].out_features == VOCAB
    assert draft.shift_label is True
    assert draft.pure_draft_prefix_len == 1


def test_domino_head_stage_freezes_backbone() -> None:
    draft = Qwen3DFlashDraftModel(_draft_config())

    draft.set_domino_training_stage("domino_head")

    head_ids = {id(parameter) for parameter in draft.domino_parameters()}
    assert head_ids
    assert all(parameter.requires_grad == (id(parameter) in head_ids) for parameter in draft.parameters())


def test_domino_warm_start_accepts_base_dflash_checkpoint(monkeypatch) -> None:
    base = Qwen3DFlashDraftModel(_draft_config(projector_type=None))
    recipe = TrainDominoRecipe.__new__(TrainDominoRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(_draft_config())
    monkeypatch.setattr(
        train_domino,
        "load_hf_safetensors_state_dict",
        lambda _path: copy.deepcopy(base.state_dict()),
    )

    recipe._load_draft_init("base-dflash")


def test_domino_warm_start_rejects_partial_domino_head(monkeypatch) -> None:
    source = Qwen3DFlashDraftModel(_draft_config())
    state_dict = copy.deepcopy(source.state_dict())
    state_dict.pop("embed_proj.2.weight")
    recipe = TrainDominoRecipe.__new__(TrainDominoRecipe)
    recipe.draft_model = Qwen3DFlashDraftModel(_draft_config())
    monkeypatch.setattr(train_domino, "load_hf_safetensors_state_dict", lambda _path: state_dict)

    with pytest.raises(ValueError, match="embed_proj.2.weight"):
        recipe._load_draft_init("partial-domino")


def test_draft_without_projector_has_no_head():
    draft = Qwen3DFlashDraftModel(_draft_config(projector_type=None))
    assert draft.projector_type is None
    assert not hasattr(draft, "prefix_gru")


def test_draft_unknown_projector_raises():
    with pytest.raises(ValueError, match="Unknown draft projector_type"):
        Qwen3DFlashDraftModel(_draft_config(projector_type="mystery"))


@pytest.mark.parametrize("shift_label", [False, True])
def test_domino_incremental_sampling_matches_training_head(shift_label: bool) -> None:
    """Autoregressive inference must use the same position mapping as the training head."""
    torch.manual_seed(23)
    trainer = _build_trainer(pure_prefix=1, shift_label=shift_label)
    draft = trainer.draft_model.eval()
    target = _TinyDominoTarget(trainer.embed_tokens, trainer.lm_head).eval()
    draft_hidden = torch.randn(1, BLOCK_SIZE, HIDDEN)
    block_ids = torch.tensor([[7, MASK_ID, MASK_ID, MASK_ID]])

    completed_ids = block_ids.clone()
    base_logits4d = target.lm_head(draft_hidden).unsqueeze(1)
    hidden4d = draft_hidden.unsqueeze(1)
    with torch.inference_mode():
        for token_position in range(1, BLOCK_SIZE):
            final_logits = trainer._apply_domino_head(
                base_logits4d=base_logits4d,
                hidden4d=hidden4d,
                prev_ids=completed_ids.unsqueeze(1),
                target_ids=completed_ids.unsqueeze(1),
            )
            head_position = token_position - 1 if shift_label else token_position
            completed_ids[:, token_position] = final_logits[:, 0, head_position, :].argmax(dim=-1)

        sampled_ids = draft._sample_draft_tokens(target, draft_hidden, block_ids)

    torch.testing.assert_close(sampled_ids, completed_ids[:, 1:])


def test_domino_sampling_uses_pure_prefix_then_causal_correction() -> None:
    """The configured pure prefix stays uncorrected while later tokens consume prior samples."""
    trainer = _build_trainer(pure_prefix=1, shift_label=True)
    draft = trainer.draft_model.eval()
    target = _TinyDominoTarget(trainer.embed_tokens, trainer.lm_head).eval()

    with torch.no_grad():
        for parameter in draft.parameters():
            parameter.zero_()
        for parameter in target.parameters():
            parameter.zero_()
        target.embed_tokens.weight[1, 0] = 1.0
        target.embed_tokens.weight[3, 0] = 1.0
        draft.prefix_gru.weight_ih_l0[2 * GRU_HIDDEN, 0] = 2.0
        draft.embed_proj[0].weight[0, HIDDEN] = 4.0
        draft.embed_proj[2].weight[3, 0] = 4.0

    draft_hidden = torch.zeros(1, BLOCK_SIZE, HIDDEN)
    block_ids = torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]])
    corrected = draft._sample_draft_tokens(target, draft_hidden, block_ids)

    torch.testing.assert_close(corrected, torch.tensor([[0, 3, 3]]))

    with torch.no_grad():
        draft.embed_proj[2].weight.zero_()
    uncorrected = draft._sample_draft_tokens(target, draft_hidden, block_ids)
    torch.testing.assert_close(uncorrected, torch.tensor([[0, 0, 0]]))


def test_domino_single_step_gru_matches_module() -> None:
    """The inference-only GRU step preserves the trained module's recurrence."""
    draft = _build_trainer().draft_model.eval()
    token_embedding = torch.randn(3, 1, HIDDEN)
    hidden_state = torch.randn(3, GRU_HIDDEN)

    expected_output, expected_hidden = draft.prefix_gru(token_embedding, hidden_state.unsqueeze(0))
    actual_output, actual_hidden = draft._domino_gru_step(token_embedding, hidden_state)

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_hidden, expected_hidden.squeeze(0))


def test_unshifted_domino_preserves_dflash_first_proposal() -> None:
    """A warm-started unshifted Domino head must not change DFlash offset 1."""
    torch.manual_seed(31)
    base = Qwen3DFlashDraftModel(_draft_config(projector_type=None)).eval()
    domino = Qwen3DFlashDraftModel(_draft_config(pure_prefix=1, shift_label=False)).eval()
    missing, unexpected = domino.load_state_dict(copy.deepcopy(base.state_dict()), strict=False)
    assert missing and all(name.startswith(("prefix_gru.", "embed_proj.")) for name in missing)
    assert not unexpected
    target = _TinyDominoTarget(nn.Embedding(VOCAB, HIDDEN), nn.Linear(HIDDEN, VOCAB, bias=False)).eval()
    draft_hidden = torch.randn(1, BLOCK_SIZE, HIDDEN)
    block_ids = torch.tensor([[7, MASK_ID, MASK_ID, MASK_ID]])

    base_suffix = base._sample_draft_tokens(target, draft_hidden, block_ids)
    domino_suffix = domino._sample_draft_tokens(target, draft_hidden, block_ids)

    torch.testing.assert_close(domino_suffix[:, :1], base_suffix[:, :1])


def test_trainer_requires_domino_draft():
    draft = Qwen3DFlashDraftModel(_draft_config(projector_type=None))
    with pytest.raises(ValueError, match="projector_type='domino'"):
        DominoTrainerModule(
            draft_model=draft,
            target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
            target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
            mask_token_id=MASK_ID,
            block_size=BLOCK_SIZE,
            attention_backend="sdpa",
        )


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shift_label", [True, False])
def test_forward_returns_metrics_and_grads_flow_to_head(shift_label):
    trainer = _build_trainer(loss_decay_gamma=7.0, shift_label=shift_label)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=0.3)
    assert isinstance(out, DominoStepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert 0.0 <= out.accuracy.item() <= 1.0
    assert 0.0 <= out.base_accuracy.item() <= 1.0
    assert out.valid_tokens.item() > 0
    assert out.loss_weight.item() > 0
    torch.testing.assert_close(out.accuracy, out.correct_tokens / out.valid_tokens)
    torch.testing.assert_close(out.base_accuracy, out.base_correct_tokens / out.valid_tokens)
    torch.testing.assert_close(out.accept_len, out.accept_len_sum / out.valid_blocks)
    torch.testing.assert_close(out.base_accept_len, out.base_accept_len_sum / out.valid_blocks)
    assert torch.isfinite(out.final_loss) and torch.isfinite(out.base_loss)
    # Accept length includes the always-accepted anchor.
    assert out.accept_len.item() >= 1.0 and out.base_accept_len.item() >= 1.0
    assert out.lambda_base.item() == pytest.approx(0.3)

    out.loss.backward()
    gru_grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.prefix_gru.parameters() if p.grad is not None)
    proj_grad = sum(
        p.grad.abs().sum().item() for p in trainer.draft_model.embed_proj.parameters() if p.grad is not None
    )
    assert gru_grad > 0
    assert proj_grad > 0


def test_forward_supports_hard_mrope_positions() -> None:
    """Domino must extend target positions to the parallel draft block."""
    draft = Qwen3DFlashDraftModel(_draft_config(shift_label=False, spatial_rope=True))
    trainer = DominoTrainerModule(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=2,
        shift_label=False,
    )
    input_ids, hidden, loss_mask = _inputs(bsz=1)
    multimodal_position_ids = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)

    output = trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        multimodal_position_ids=multimodal_position_ids,
    )

    assert torch.isfinite(output.loss)


def test_lambda_base_one_equals_base_loss():
    trainer = _build_trainer()
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=1.0)
    assert out.loss.item() == pytest.approx(out.base_loss.item(), rel=1e-5)


def test_lambda_base_zero_equals_final_loss():
    trainer = _build_trainer()
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=0.0)
    assert out.loss.item() == pytest.approx(out.final_loss.item(), rel=1e-5)


def test_suffix_start_depends_on_shift_label():
    # shift_label=True -> pure_prefix; shift_label=False -> 1 + pure_prefix.
    assert _build_trainer(pure_prefix=1, shift_label=True)._suffix_start == 1
    assert _build_trainer(pure_prefix=1, shift_label=False)._suffix_start == 2
    assert _build_trainer(pure_prefix=2, shift_label=True)._suffix_start == 2


# --------------------------------------------------------------------------- #
# Curriculum schedule + acceptance length
# --------------------------------------------------------------------------- #


def test_get_lambda_base_schedule():
    # Linear decay from start to 0 over decay_ratio * total steps, then clamps at 0.
    assert get_lambda_base(0, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(1.0)
    assert get_lambda_base(50, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(0.5)
    assert get_lambda_base(100, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(0.0)
    assert get_lambda_base(200, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(0.0)
    # Half-decay: lambda_base hits 0 at the midpoint.
    assert get_lambda_base(50, 100, lambda_start=1.0, decay_ratio=0.5) == pytest.approx(0.0)
    assert get_lambda_base(0, 0, lambda_start=1.0, decay_ratio=0.5) == pytest.approx(1.0)


def test_compute_accept_len():
    # block 0: first two predictions correct then a miss -> accept_len 2.
    # block 1: first prediction wrong -> accept_len 0.
    pred = torch.tensor([[[1, 2, 9, 4], [9, 2, 3, 4]]])
    target = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    accept = compute_accept_len(pred, target, valid)
    assert accept.tolist() == [[2.0, 0.0]]
    # An invalid trailing position never truncates the accepted prefix.
    valid2 = torch.tensor([[[True, True, False, True], [True, True, True, True]]])
    pred2 = torch.tensor([[[1, 2, 0, 4], [1, 2, 3, 4]]])
    accept2 = compute_accept_len(pred2, target, valid2)
    assert accept2.tolist() == [[3.0, 4.0]]
