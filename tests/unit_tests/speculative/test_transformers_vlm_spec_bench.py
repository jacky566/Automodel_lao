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

"""Tests for the batch-one Transformers speculative benchmark."""

import json
import logging
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tools.transformers_vlm_spec_bench import (
    FIXED_NEW_TOKENS,
    REPRESENTATIVE_BENCHMARKS,
    _acceptance_lengths,
    _greedy_cached_forward,
    _load_baseline_throughput,
    _load_benchmark_specs,
    _load_sharegpt_vlm_prompts,
    _report_sample_progress,
)


class _TinyPositionModel(nn.Module):
    """Return shape-compatible multimodal position IDs for cached decoding."""

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
        **kwargs,
    ) -> torch.Tensor:
        """Build position IDs for one incremental token.

        Args:
            input_ids: Tensor of shape [1, query_sequence].
            inputs_embeds: Tensor of shape [1, query_sequence, hidden].
            attention_mask: Tensor of shape [1, prefix_sequence + query_sequence].
            past_key_values: Target cache covering ``prefix_sequence`` tokens.
            **kwargs: Unused multimodal position metadata.

        Returns:
            Tensor of shape [3, 1, prefix_sequence + query_sequence].
        """
        del input_ids, inputs_embeds, past_key_values, kwargs
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        return positions.view(1, 1, -1).expand(3, 1, -1)


class _TinyCachedTarget(nn.Module):
    """Deterministic target whose next token is the current token plus one."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyPositionModel()
        self.embeddings = nn.Embedding(8, 4)
        self.query_lengths: list[int] = []
        self.cache_ids: list[int] = []
        self.cache_positions: list[list[int] | None] = []
        self.position_shapes: list[tuple[int, ...] | None] = []

    def get_input_embeddings(self) -> nn.Module:
        """Return the target token embeddings."""
        return self.embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        *,
        next_sequence_length: int,
        past_key_values,
        attention_mask: torch.Tensor,
        is_first_iteration: bool,
        use_cache: bool,
        **kwargs,
    ) -> dict[str, object]:
        """Slice the uncached query tokens from a full sequence.

        Args:
            input_ids: Tensor of shape [1, full_sequence].
            next_sequence_length: Number of trailing query tokens.
            past_key_values: Target cache covering the processed prefix.
            attention_mask: Tensor of shape [1, full_sequence].
            is_first_iteration: Whether this is the prompt prefill.
            use_cache: Whether the target should update its cache.
            **kwargs: Unused processor tensors.

        Returns:
            Mapping containing ``input_ids`` of shape [1, query_sequence], the
            full attention mask, and the persistent target cache.
        """
        del is_first_iteration, kwargs
        return {
            "input_ids": input_ids[:, -next_sequence_length:],
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Append query tokens to the cache and return successor logits.

        Args:
            input_ids: Tensor of shape [1, query_sequence].
            past_key_values: Target cache mutated to append ``query_sequence`` tokens.
            attention_mask: Tensor of shape [1, processed_sequence + query_sequence].
            cache_position: Tensor of shape [query_sequence], or ``None`` during prefill.
            position_ids: Tensor of shape [4, 1, query_sequence] during cached decode.
            **kwargs: Unused target forward arguments.

        Returns:
            Object containing logits of shape [1, query_sequence, vocab] and the
            mutated target cache.
        """
        del attention_mask, kwargs
        query_length = input_ids.shape[1]
        cache_tensor = torch.zeros((1, 1, query_length, 1), device=input_ids.device)
        past_key_values.update(cache_tensor, cache_tensor, 0)
        self.query_lengths.append(query_length)
        self.cache_ids.append(id(past_key_values))
        self.cache_positions.append(None if cache_position is None else cache_position.tolist())
        self.position_shapes.append(None if position_ids is None else tuple(position_ids.shape))
        logits = torch.full((1, query_length, 8), -1000.0, device=input_ids.device)
        logits.scatter_(-1, input_ids.add(1).remainder(8).unsqueeze(-1), 0.0)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_greedy_cached_forward_reuses_cache_and_decodes_one_token_at_a_time() -> None:
    """The target prefills once and then uses one-token cached forwards."""
    target = _TinyCachedTarget()

    generated = _greedy_cached_forward(
        target,
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        max_new_tokens=6,
        eos_token_id=5,
    )

    assert generated == [3, 4, 5]
    assert target.query_lengths == [2, 1, 1]
    assert len(set(target.cache_ids)) == 1
    assert target.cache_positions == [None, [2], [3]]
    assert target.position_shapes == [None, (4, 1, 1), (4, 1, 1)]


def test_acceptance_lengths_separates_accepted_draft_from_emitted_tokens() -> None:
    """Accept length excludes the one guaranteed target token emitted per round."""
    assert _acceptance_lengths(9, 4) == (2.25, 3.25)
    assert _acceptance_lengths(0, 0) == (None, None)


def test_representative_suite_has_four_benchmarks_with_fixed_512_token_outputs() -> None:
    """The default suite spans four task types with an identical generation length."""
    specs = _load_benchmark_specs()

    assert tuple(spec["name"] for spec in specs) == REPRESENTATIVE_BENCHMARKS
    assert len(specs) == 4
    assert {spec["max_new_tokens"] for spec in specs} == {FIXED_NEW_TOKENS}


def test_sample_progress_reports_benchmark_and_position(caplog) -> None:
    """Each completed sample identifies its mode, benchmark, and position."""
    with caplog.at_level(logging.INFO):
        _report_sample_progress(
            mode="dflash",
            benchmark_name="scienceqa",
            sample_index=7,
            total_samples=100,
            output_tokens=7 * 512,
            start_time=time.perf_counter() - 1.0,
        )

    assert "[dflash] benchmark=scienceqa sample=7/100" in caplog.text
    assert "rolling_tok_s=" in caplog.text


def test_load_baseline_throughput_validates_matching_run(tmp_path) -> None:
    """A saved baseline supplies throughput without storing or regenerating token IDs."""
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "scienceqa": {
                    "mode": "baseline",
                    "num_prompts": 100,
                    "max_new_tokens": 512,
                    "fixed_output_length": True,
                    "tok_s": 50.0,
                }
            }
        )
    )

    assert _load_baseline_throughput(
        path,
        {"scienceqa": 512},
        num_prompts=100,
        fixed_output_length=True,
    ) == {"scienceqa": 50.0}
    with pytest.raises(ValueError, match="current run uses 20"):
        _load_baseline_throughput(
            path,
            {"scienceqa": 512},
            num_prompts=20,
            fixed_output_length=True,
        )


def test_dflash_main_does_not_run_an_internal_baseline(monkeypatch, tmp_path) -> None:
    """DFlash performs only speculative decoding unless a saved baseline is supplied."""
    from tools import transformers_vlm_spec_bench as benchmark

    output = tmp_path / "dflash.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transformers_vlm_spec_bench.py",
            "--target",
            "target",
            "--draft",
            "draft",
            "--mode",
            "dflash",
            "--num-prompts",
            "1",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "_load_benchmark_specs",
        lambda: [{"name": "scienceqa"}],
    )
    monkeypatch.setattr(benchmark, "_load_official_prompts", lambda spec, count: [[{"role": "user"}]])
    monkeypatch.setattr(benchmark, "_load_target", lambda *args: object())
    monkeypatch.setattr(benchmark.AutoProcessor, "from_pretrained", lambda *args: object())
    monkeypatch.setattr(benchmark, "_load_dflash", lambda *args: object())
    monkeypatch.setattr(benchmark, "_baseline", lambda *args, **kwargs: pytest.fail("unexpected baseline"))
    monkeypatch.setattr(benchmark, "_dflash", lambda *args, **kwargs: {"tok_s": 75.0})

    benchmark.main()

    assert json.loads(output.read_text())["scienceqa"]["mode"] == "dflash"


def test_load_sharegpt_vlm_prompts_preserves_image_order(tmp_path) -> None:
    """Local benchmark prompts retain the source placeholder ordering."""
    media_dir = tmp_path / "images"
    media_dir.mkdir()
    (media_dir / "first.jpg").touch()
    (media_dir / "second.jpg").touch()
    input_data = tmp_path / "data.jsonl"
    rows = [
        {
            "conversations": [{"from": "human", "value": "skip <image>"}],
            "images": ["first.jpg"],
        },
        {
            "conversations": [{"from": "human", "value": "before <image> middle <image> after"}],
            "images": ["first.jpg", "second.jpg"],
        },
    ]
    input_data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    prompts = _load_sharegpt_vlm_prompts(input_data, media_dir, start=1, limit=1)

    assert prompts == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": str(media_dir / "first.jpg")}},
                    {"type": "text", "text": "middle"},
                    {"type": "image_url", "image_url": {"url": str(media_dir / "second.jpg")}},
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    ]


def test_load_sharegpt_vlm_prompts_rejects_placeholder_mismatch(tmp_path) -> None:
    """Malformed multimodal rows fail before loading the target model."""
    input_data = tmp_path / "data.jsonl"
    input_data.write_text('{"conversations":[{"from":"human","value":"<image>"}],"images":[]}\n')

    with pytest.raises(ValueError, match="1 image placeholders but 0 image paths"):
        _load_sharegpt_vlm_prompts(input_data, tmp_path, start=0, limit=1)
