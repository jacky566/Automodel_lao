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

"""Batch-one Transformers evaluator for Qwen2.5-VL speculative drafts.

The evaluator deliberately uses the same prompt adapter and generation limits
for baseline, DFlash and ViSpec. The ViSpec path flattens each candidate tree
into one target forward, then compacts the accepted path in the target KV cache.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, PretrainedConfig
from transformers.cache_utils import DynamicCache

FIXED_NEW_TOKENS = 512
LEGACY_NEW_TOKENS = 256
REPRESENTATIVE_BENCHMARKS = (
    "scienceqa",
    "mmvet",
    "textvqa",
    "coco_caption",
)
BENCHMARK_CONFIG = Path(__file__).parents[1] / "examples/speculative/bench_sweep/vlm_spec_bench_datasets.yaml"
LEGACY_BENCHMARK_CONFIG = (
    Path(__file__).parents[1] / "examples/speculative/bench_sweep/vlm_spec_bench_datasets_legacy_256.yaml"
)
LEGACY_BENCHMARKS = (
    "gqa",
    "textvqa",
    "coco_caption",
    "charxiv_reasoning",
    "mmmu_pro",
)
TEXT_BENCHMARK_CONFIG = Path(__file__).parents[1] / "examples/speculative/bench_sweep/text_spec_bench_datasets_256.yaml"
TEXT_BENCHMARKS = ("mt_bench", "humaneval", "gsm8k", "alpaca")
VISPEC_DEPTH = 3
VISPEC_TOP_K = 8
VISPEC_TOTAL_TOKEN = 30
POSITION_BUCKETS = ((1, 32), (33, 64), (65, 128), (129, 256), (257, 512))
logger = logging.getLogger(__name__)


def _load_benchmark_specs(
    path: Path = BENCHMARK_CONFIG,
    expected_benchmarks: tuple[str, ...] = REPRESENTATIVE_BENCHMARKS,
    expected_max_new_tokens: int = FIXED_NEW_TOKENS,
) -> list[dict[str, Any]]:
    """Load and validate an ordered VLM benchmark suite."""
    payload = yaml.safe_load(path.read_text())
    specs = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(specs, list):
        raise ValueError(f"Benchmark config must contain a datasets list: {path}")
    names = tuple(spec.get("name") for spec in specs if isinstance(spec, dict))
    if names != expected_benchmarks:
        raise ValueError(f"Expected VLM benchmarks {expected_benchmarks}, got {names}.")
    invalid_lengths = [spec.get("name") for spec in specs if spec.get("max_new_tokens") != expected_max_new_tokens]
    if invalid_lengths:
        raise ValueError(f"Official benchmarks must all generate {expected_max_new_tokens} tokens: {invalid_lengths}")
    return specs


def _load_hf_rows(
    input_data: str,
    *,
    split: str,
    name: str | None,
    shuffle_seed: int | None,
):
    """Stream rows from Hugging Face without downloading complete benchmark corpora."""
    from datasets import load_dataset

    rows = load_dataset(input_data, name=name, split=split, streaming=True)
    if shuffle_seed is not None:
        rows = rows.shuffle(seed=shuffle_seed, buffer_size=1_000)
    return rows


def _load_official_prompts(spec: dict[str, Any], num_prompts: int) -> list[list[dict[str, Any]]]:
    """Load one official benchmark and adapt it to OpenAI Vision messages."""
    from nemo_automodel.components.speculative.bench_multimodal import load_multimodal_prompts

    prompt_args = argparse.Namespace(
        benchmark_adapter=spec["benchmark_adapter"],
        input_data=spec["input_data"],
        split=spec.get("split", "test"),
        dataset_name=spec.get("dataset_name"),
        shuffle_seed=None,
        num_prompts=num_prompts,
    )
    prompts = load_multimodal_prompts(prompt_args, _load_hf_rows)
    if len(prompts) != num_prompts:
        raise ValueError(
            f"Benchmark {spec['name']} provided {len(prompts)} valid prompts; {num_prompts} were requested."
        )
    return prompts


def _load_text_prompts(spec: dict[str, Any], num_prompts: int) -> list[list[dict[str, Any]]]:
    """Load deterministic single-turn prompts from a text benchmark."""
    rows = _load_hf_rows(
        spec["input_data"],
        split=spec.get("split", "test"),
        name=spec.get("dataset_name"),
        shuffle_seed=None,
    )
    prompts: list[list[dict[str, Any]]] = []
    prompt_column = spec["prompt_column"]
    context_column = spec.get("prompt_context_column")
    for row in rows:
        prompt = row.get(prompt_column)
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else None
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        context = row.get(context_column) if context_column is not None else None
        if isinstance(context, str) and context.strip():
            prompt = f"{prompt.rstrip()}\n\n{context.strip()}"
        prompts.append([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        if len(prompts) == num_prompts:
            break
    if len(prompts) != num_prompts:
        raise ValueError(
            f"Benchmark {spec['name']} provided {len(prompts)} valid prompts; {num_prompts} were requested."
        )
    return prompts


def _report_sample_progress(
    *,
    mode: str,
    benchmark_name: str,
    sample_index: int,
    total_samples: int,
    output_tokens: int,
    start_time: float,
) -> None:
    """Log sample-level progress and rolling output throughput."""
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    logger.info(
        "[%s] benchmark=%s sample=%d/%d completed elapsed=%.1fs rolling_tok_s=%.2f",
        mode,
        benchmark_name,
        sample_index,
        total_samples,
        elapsed,
        output_tokens / elapsed,
    )


def _load_baseline_throughput(
    path: Path,
    benchmark_limits: dict[str, int],
    *,
    num_prompts: int,
    fixed_output_length: bool,
) -> dict[str, float]:
    """Load compatible per-benchmark throughput from a completed baseline run."""
    payload = json.loads(path.read_text())
    throughputs: dict[str, float] = {}
    for benchmark_name, max_new_tokens in benchmark_limits.items():
        result = payload.get(benchmark_name) if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            raise ValueError(f"Baseline results do not contain benchmark {benchmark_name!r}: {path}")
        if result.get("mode") != "baseline":
            raise ValueError(f"Baseline result for {benchmark_name!r} has mode={result.get('mode')!r}.")
        if result.get("num_prompts") != num_prompts:
            raise ValueError(
                f"Baseline result for {benchmark_name!r} used {result.get('num_prompts')} prompts; "
                f"the current run uses {num_prompts}."
            )
        if result.get("max_new_tokens") != max_new_tokens:
            raise ValueError(
                f"Baseline result for {benchmark_name!r} used max_new_tokens={result.get('max_new_tokens')}; "
                f"the current run uses {max_new_tokens}."
            )
        if bool(result.get("fixed_output_length")) != fixed_output_length:
            raise ValueError(f"Baseline result for {benchmark_name!r} used a different fixed-output-length mode.")
        throughput = result.get("tok_s")
        if not isinstance(throughput, (int, float)) or throughput <= 0:
            raise ValueError(f"Baseline result for {benchmark_name!r} has invalid tok_s={throughput!r}.")
        throughputs[benchmark_name] = float(throughput)
    return throughputs


def _load_sharegpt_vlm_prompts(
    input_data: Path,
    media_dir: Path,
    *,
    start: int,
    limit: int,
) -> list[list[dict[str, Any]]]:
    """Load deterministic multimodal prompts from a ShareGPT-style JSONL.

    Args:
        input_data: JSONL whose rows contain ``conversations`` and ``images``.
        media_dir: Directory used to resolve relative image paths.
        start: Zero-based source row at which to begin.
        limit: Exact number of prompts to load.

    Returns:
        OpenAI Vision-style user messages with image placeholders expanded in
        their original text order.
    """
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}.")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}.")

    prompts: list[list[dict[str, Any]]] = []
    with input_data.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index < start:
                continue
            if len(prompts) == limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {input_data}:{row_index + 1}.") from error
            conversations = row.get("conversations")
            images = row.get("images")
            if not isinstance(conversations, list) or not isinstance(images, list):
                raise ValueError(f"Row {row_index} must contain list-valued conversations and images.")
            user_turn = next(
                (turn for turn in conversations if isinstance(turn, dict) and turn.get("from") in {"human", "user"}),
                None,
            )
            if user_turn is None or not isinstance(user_turn.get("value"), str):
                raise ValueError(f"Row {row_index} has no string-valued human conversation turn.")
            text_parts = user_turn["value"].split("<image>")
            if len(text_parts) - 1 != len(images):
                raise ValueError(
                    f"Row {row_index} has {len(text_parts) - 1} image placeholders but {len(images)} image paths."
                )
            content: list[dict[str, Any]] = []
            for image_index, text_part in enumerate(text_parts):
                if text_part.strip():
                    content.append({"type": "text", "text": text_part.strip()})
                if image_index < len(images):
                    image_path = Path(images[image_index])
                    absolute_image = image_path if image_path.is_absolute() else media_dir / image_path
                    if not absolute_image.is_file():
                        raise FileNotFoundError(f"Image for row {row_index} does not exist: {absolute_image}")
                    content.append({"type": "image_url", "image_url": {"url": str(absolute_image)}})
            prompts.append([{"role": "user", "content": content}])

    if len(prompts) != limit:
        raise ValueError(
            f"Requested {limit} prompts starting at row {start}, but {input_data} provided {len(prompts)}."
        )
    return prompts


def _to_hf_messages(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI Vision messages into Transformers chat-template messages."""
    messages: list[dict[str, Any]] = []
    for message in prompt:
        content = message.get("content", [])
        hf_content: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "text":
                hf_content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                hf_content.append({"type": "image", "image": part["image_url"]["url"]})
        messages.append({"role": message["role"], "content": hf_content})
    return messages


def _prepare_inputs(processor, prompt: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Build one Qwen-VL processor batch with batch axis one."""
    messages = _to_hf_messages(prompt)
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}


def _load_target(path: str, device: torch.device, attn_implementation: str = "eager"):
    return (
        AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation=attn_implementation,
        )
        .to(device)
        .eval()
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _acceptance_lengths(accepted_tokens: int | float, verify_steps: int | float) -> tuple[float | None, float | None]:
    """Return accepted draft tokens and actual emitted tokens per round."""
    if verify_steps == 0:
        return None, None
    accept_length = accepted_tokens / verify_steps
    return accept_length, 1.0 + accept_length


def _aggregate_position_acceptance(
    events: list[dict[str, int]],
    max_new_tokens: int,
) -> dict[str, dict[str, int | float | None]]:
    """Aggregate verification acceptance by one-based round-start position.

    Each speculative verification round is assigned to the bucket containing
    its anchor's generated-token position. This preserves the existing
    accepted-tokens-per-verification-step definition within every bucket.

    Args:
        events: Per-round position, accepted-token, and draft-token counters.
        max_new_tokens: Maximum number of generated tokens in each sample.

    Returns:
        Position labels mapped to verification counts and acceptance metrics.
    """
    aggregated: dict[str, dict[str, int | float | None]] = {}
    for lower, upper in POSITION_BUCKETS:
        if lower > max_new_tokens:
            continue
        clipped_upper = min(upper, max_new_tokens)
        bucket_events = [event for event in events if lower <= event["generated_position"] <= clipped_upper]
        verify_steps = len(bucket_events)
        accepted_tokens = sum(event["accepted_tokens"] for event in bucket_events)
        draft_tokens = sum(event["draft_tokens"] for event in bucket_events)
        accept_length, emitted_tokens_per_step = _acceptance_lengths(accepted_tokens, verify_steps)
        aggregated[f"{lower}-{clipped_upper}"] = {
            "verify_steps": verify_steps,
            "accepted_tokens": accepted_tokens,
            "draft_tokens": draft_tokens,
            "accept_length": accept_length,
            "emitted_tokens_per_step": emitted_tokens_per_step,
            "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        }
    return aggregated


def _aggregate_proposal_offset_acceptance(
    events: list[dict[str, int]],
) -> dict[str, dict[str, int | float | None]]:
    """Aggregate prefix acceptance for each draft offset after the anchor.

    Args:
        events: Per-round accepted-token and in-range draft-token counters. An
            event with ``accepted_tokens >= k`` accepted proposal offset ``k``.

    Returns:
        One-based proposal offsets mapped to opportunities, accepted counts,
        and acceptance rates.
    """
    max_offset = max((event["draft_tokens"] for event in events), default=0)
    aggregated: dict[str, dict[str, int | float | None]] = {}
    for offset in range(1, max_offset + 1):
        opportunities = sum(event["draft_tokens"] >= offset for event in events)
        accepted = sum(event["accepted_tokens"] >= offset for event in events)
        aggregated[str(offset)] = {
            "opportunities": opportunities,
            "accepted": accepted,
            "acceptance_rate": accepted / opportunities if opportunities else None,
        }
    return aggregated


def _greedy_cached_forward(
    target: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    """Decode greedily with one persistent target KV cache.

    Args:
        target: Transformers image-text model whose forward method accepts a
            ``DynamicCache`` and Qwen-style multimodal position IDs.
        model_inputs: Processor tensors containing ``input_ids`` and
            ``attention_mask`` of shape [1, prompt_sequence]. Vision tensors
            retain the processor-defined flattened patch layouts.
        max_new_tokens: Maximum number of tokens to generate after the prompt.
        eos_token_id: Token that terminates decoding after it is emitted, or
            ``None`` to decode exactly ``max_new_tokens`` tokens.

    Returns:
        Generated token IDs. The prompt is excluded.
    """
    if max_new_tokens < 1:
        return []
    input_ids = model_inputs.get("input_ids")
    prompt_attention_mask = model_inputs.get("attention_mask")
    if input_ids is None or prompt_attention_mask is None:
        raise ValueError("Cached target decoding requires input_ids and attention_mask.")
    if input_ids.shape[0] != 1 or prompt_attention_mask.shape != input_ids.shape:
        raise ValueError("Cached target decoding requires matching batch-one input ids and attention mask.")

    prompt_length = input_ids.shape[1]
    cache = DynamicCache()
    target_kwargs = {key: value for key, value in model_inputs.items() if key != "input_ids"}
    prefill_kwargs = target.prepare_inputs_for_generation(
        input_ids,
        next_sequence_length=prompt_length,
        past_key_values=cache,
        is_first_iteration=True,
        use_cache=True,
        **target_kwargs,
    )
    prefill_input_ids = prefill_kwargs.pop("input_ids")
    prefill_kwargs["logits_to_keep"] = 1
    prefill_kwargs["return_dict"] = True
    outputs = target(prefill_input_ids, **prefill_kwargs)
    cache = outputs.past_key_values
    if cache is None or cache.get_seq_length() != prompt_length:
        actual_length = None if cache is None else cache.get_seq_length()
        raise RuntimeError(f"Target prefill cache must cover {prompt_length} prompt tokens, got {actual_length}.")

    generated_ids = [int(outputs.logits[:, -1].argmax(dim=-1).item())]
    target_forward_params = inspect.signature(target.forward).parameters
    position_hook = getattr(getattr(target, "model", None), "compute_3d_position_ids", None)
    if not callable(position_hook):
        raise RuntimeError("Cached target decoding requires compute_3d_position_ids on the target base model.")
    position_hook_params = inspect.signature(position_hook).parameters

    while len(generated_ids) < max_new_tokens and (eos_token_id is None or generated_ids[-1] != eos_token_id):
        token_ids = torch.tensor([generated_ids[-1:]], dtype=input_ids.dtype, device=input_ids.device)
        active_length = prompt_length + len(generated_ids)
        full_attention_mask = torch.cat(
            (prompt_attention_mask, prompt_attention_mask.new_ones((1, len(generated_ids)))), dim=1
        )
        full_input_ids = torch.cat(
            (input_ids, torch.tensor([generated_ids], dtype=input_ids.dtype, device=input_ids.device)), dim=1
        )
        generation_kwargs = {
            key: value
            for key, value in target_kwargs.items()
            if key not in {"attention_mask", "mm_token_type_ids", "pixel_values", "pixel_values_videos"}
        }
        step_kwargs = target.prepare_inputs_for_generation(
            full_input_ids,
            next_sequence_length=1,
            past_key_values=cache,
            attention_mask=full_attention_mask,
            is_first_iteration=False,
            use_cache=True,
            **generation_kwargs,
        )
        step_input_ids = step_kwargs.pop("input_ids")
        step_kwargs["attention_mask"] = full_attention_mask
        position_kwargs: dict[str, object] = {
            "input_ids": token_ids,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "inputs_embeds": target.get_input_embeddings()(token_ids),
            "attention_mask": full_attention_mask,
            "past_key_values": cache,
            "second_per_grid_ts": None,
            "mm_token_type_ids": None,
        }
        multimodal_position_ids = position_hook(
            **{key: value for key, value in position_kwargs.items() if key in position_hook_params}
        )
        text_position_ids = full_attention_mask.long().cumsum(dim=-1) - 1
        text_position_ids.masked_fill_(full_attention_mask == 0, 0)
        step_kwargs["position_ids"] = torch.cat(
            (text_position_ids.unsqueeze(0), multimodal_position_ids),
            dim=0,
        )[..., -1:]
        if "cache_position" in target_forward_params:
            step_kwargs["cache_position"] = torch.tensor([active_length - 1], dtype=torch.long, device=input_ids.device)
        step_kwargs["logits_to_keep"] = 1
        step_kwargs["return_dict"] = True
        outputs = target(step_input_ids, **step_kwargs)
        cache = outputs.past_key_values
        if cache is None or cache.get_seq_length() != active_length:
            actual_length = None if cache is None else cache.get_seq_length()
            raise RuntimeError(f"Target decode cache must cover {active_length} processed tokens, got {actual_length}.")
        generated_ids.append(int(outputs.logits[:, -1].argmax(dim=-1).item()))
    return generated_ids


@torch.inference_mode()
def _baseline(
    target,
    processor,
    prompts: list[list[dict[str, Any]]],
    max_new_tokens: int,
    device: torch.device,
    fixed_output_length: bool = False,
    *,
    benchmark_name: str = "benchmark",
) -> dict[str, Any]:
    """Run one-token cached target forwards and report output-token throughput."""
    output_tokens = 0
    reference_outputs: list[list[int]] = []
    eos_token_id = None if fixed_output_length else getattr(processor.tokenizer, "eos_token_id", None)
    if prompts:
        warmup_inputs = _prepare_inputs(processor, prompts[0], device)
        _greedy_cached_forward(
            target,
            warmup_inputs,
            max_new_tokens=min(8, max_new_tokens),
            eos_token_id=eos_token_id,
        )
    _sync(device)
    start = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        logger.info(
            "[baseline] benchmark=%s sample=%d/%d started",
            benchmark_name,
            prompt_index + 1,
            len(prompts),
        )
        inputs = _prepare_inputs(processor, prompt, device)
        generated = _greedy_cached_forward(
            target,
            inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        if fixed_output_length and len(generated) != max_new_tokens:
            raise RuntimeError(f"Baseline generated {len(generated)} tokens; expected exactly {max_new_tokens}.")
        output_tokens += len(generated)
        reference_outputs.append(generated)
        _report_sample_progress(
            mode="baseline",
            benchmark_name=benchmark_name,
            sample_index=prompt_index + 1,
            total_samples=len(prompts),
            output_tokens=output_tokens,
            start_time=start,
        )
    _sync(device)
    wall = time.perf_counter() - start
    return {
        "completed": len(prompts),
        "output_tokens": output_tokens,
        "wall_clock_s": wall,
        "tok_s": output_tokens / wall,
        "_reference_outputs": reference_outputs,
    }


def _scale_visual_gate_state_dict(state_dict: dict[str, torch.Tensor], multiplier: float) -> None:
    """Scale the pooled-MLP visual gates in a loaded DFlash state dict.

    Args:
        state_dict: Mapping of checkpoint names to tensors of arbitrary shape. Entries
            ending in ``.visual_fusion.gate`` must be scalar tensors and are replaced
            with independently scaled tensors; all other entries remain unchanged.
        multiplier: Non-negative finite scale applied to every visual gate.

    Raises:
        ValueError: If the multiplier is invalid or the state dict has no visual gates.
    """
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError(f"visual gate multiplier must be finite and non-negative, got {multiplier}.")
    gate_keys = [key for key in state_dict if key.endswith(".visual_fusion.gate")]
    if not gate_keys:
        raise ValueError("visual gate multiplier requires a DFlash checkpoint with visual_fusion.gate tensors.")
    for key in gate_keys:
        if state_dict[key].numel() != 1:
            raise ValueError(f"Expected scalar visual gate tensor for {key}, got shape {tuple(state_dict[key].shape)}.")
        state_dict[key] = state_dict[key] * multiplier


def _load_dflash(
    path: str,
    target,
    device: torch.device,
    *,
    visual_gate_multiplier: float = 1.0,
):
    from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

    config = AutoConfig.from_pretrained(path)
    draft = Qwen3DFlashDraftModel(config)
    from safetensors.torch import load_file

    checkpoint_dir = Path(path)
    weight_files = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found under {checkpoint_dir}")
    state_dict = {}
    for weight_file in weight_files:
        state_dict.update(load_file(str(weight_file)))
    if visual_gate_multiplier != 1.0:
        _scale_visual_gate_state_dict(state_dict, visual_gate_multiplier)
    draft.load_state_dict(state_dict, strict=True)
    return draft.to(device=device, dtype=next(target.parameters()).dtype).eval()


@torch.inference_mode()
def _dflash(
    target,
    processor,
    draft,
    prompts,
    max_new_tokens: int,
    device: torch.device,
    reference_outputs=None,
    fixed_output_length: bool = False,
    verification_mode: str = "block",
    *,
    benchmark_name: str = "benchmark",
    draft_image_context_mode: str = "keep",
) -> dict[str, Any]:
    """Run DFlash's block verifier with prompt-only multimodal target inputs."""
    output_tokens = 0
    draft_tokens = 0.0
    accepted_tokens = 0.0
    verify_steps = 0.0
    exact_matches = 0
    matching_tokens = 0
    compared_tokens = 0
    common_prefix_tokens = 0
    target_prefill_seconds = 0.0
    draft_seconds = 0.0
    target_verify_seconds = 0.0
    acceptance_events: list[dict[str, int]] = []
    draft_image_token_id = int(getattr(target.config, "image_token_id")) if draft_image_context_mode != "keep" else None
    if prompts:
        warmup_inputs = _prepare_inputs(processor, prompts[0], device)
        warmup_ids = warmup_inputs.pop("input_ids")
        draft.spec_generate(
            target,
            warmup_ids,
            min(8, max_new_tokens),
            stop_token_ids=None,
            temperature=0.0,
            target_kwargs=warmup_inputs,
            sequential_target_verification=verification_mode == "sequential",
            draft_image_context_mode=draft_image_context_mode,
            draft_image_token_id=draft_image_token_id,
        )
    _sync(device)
    start = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        logger.info(
            "[dflash] benchmark=%s sample=%d/%d started",
            benchmark_name,
            prompt_index + 1,
            len(prompts),
        )
        inputs = _prepare_inputs(processor, prompt, device)
        prompt_ids = inputs.pop("input_ids")
        sample_max_new_tokens = (
            len(reference_outputs[prompt_index])
            if fixed_output_length and reference_outputs is not None
            else max_new_tokens
        )
        output, stats = draft.spec_generate(
            target,
            prompt_ids,
            sample_max_new_tokens,
            stop_token_ids=None
            if fixed_output_length
            else (
                [int(processor.tokenizer.eos_token_id)]
                if getattr(processor.tokenizer, "eos_token_id", None) is not None
                else None
            ),
            temperature=0.0,
            target_kwargs=inputs,
            return_stats=True,
            sequential_target_verification=verification_mode == "sequential",
            draft_image_context_mode=draft_image_context_mode,
            draft_image_token_id=draft_image_token_id,
        )
        generated_length = int(output.shape[1] - prompt_ids.shape[1])
        if fixed_output_length and generated_length != sample_max_new_tokens:
            raise RuntimeError(f"DFlash generated {generated_length} tokens; expected exactly {sample_max_new_tokens}.")
        output_tokens += generated_length
        if reference_outputs is not None:
            generated_ids = output[0, prompt_ids.shape[1] :].tolist()
            reference_ids = reference_outputs[prompt_index]
            exact_matches += int(generated_ids == reference_ids)
            compared_length = min(len(generated_ids), len(reference_ids))
            matching_tokens += sum(
                generated_ids[token_index] == reference_ids[token_index] for token_index in range(compared_length)
            )
            compared_tokens += max(len(generated_ids), len(reference_ids))
            for generated_id, reference_id in zip(generated_ids, reference_ids):
                if generated_id != reference_id:
                    break
                common_prefix_tokens += 1
        draft_tokens += stats["draft_tokens"]
        accepted_tokens += stats["accepted_tokens"]
        verify_steps += stats["verify_steps"]
        acceptance_events.extend(stats["acceptance_events"])
        target_prefill_seconds += stats["target_prefill_seconds"]
        draft_seconds += stats["draft_seconds"]
        target_verify_seconds += stats["target_verify_seconds"]
        _report_sample_progress(
            mode="dflash",
            benchmark_name=benchmark_name,
            sample_index=prompt_index + 1,
            total_samples=len(prompts),
            output_tokens=output_tokens,
            start_time=start,
        )
    _sync(device)
    wall = time.perf_counter() - start
    accept_length, emitted_tokens_per_step = _acceptance_lengths(accepted_tokens, verify_steps)
    return {
        "completed": len(prompts),
        "output_tokens": output_tokens,
        "wall_clock_s": wall,
        "tok_s": output_tokens / wall,
        "accept_length": accept_length,
        "emitted_tokens_per_step": emitted_tokens_per_step,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        "position_acceptance_by_round_start": _aggregate_position_acceptance(
            acceptance_events,
            max_new_tokens,
        ),
        "proposal_offset_acceptance": _aggregate_proposal_offset_acceptance(acceptance_events),
        "exact_match_count": exact_matches if reference_outputs is not None else None,
        "exact_match_rate": exact_matches / len(prompts) if reference_outputs is not None and prompts else None,
        "token_match_rate": matching_tokens / compared_tokens
        if reference_outputs is not None and compared_tokens
        else None,
        "mean_common_prefix_length": common_prefix_tokens / len(prompts)
        if reference_outputs is not None and prompts
        else None,
        "target_prefill_s": target_prefill_seconds,
        "draft_s": draft_seconds,
        "target_verify_s": target_verify_seconds,
        "unattributed_s": max(0.0, wall - target_prefill_seconds - draft_seconds - target_verify_seconds),
    }


def _load_vispec(path: str, target, device: torch.device):
    from nemo_automodel.components.speculative.eagle.vispec_draft import VispecDraftModel

    config = PretrainedConfig.from_dict(json.loads((Path(path) / "config.json").read_text()))
    if getattr(config, "architectures", []) != ["VispecDraftModel"]:
        raise ValueError(
            "ViSpec evaluator expects a NeMo consolidated checkpoint with architectures=['VispecDraftModel']; "
            "the official JLKang checkpoint uses the original ViSpec key layout and needs a conversion step."
        )
    draft = VispecDraftModel(config)
    from safetensors.torch import load_file

    checkpoint_dir = Path(path)
    weight_files = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found under {checkpoint_dir}")
    state_dict = {}
    for weight_file in weight_files:
        state_dict.update(load_file(str(weight_file)))
    draft.load_state_dict(state_dict, strict=True)
    return draft.to(device=device, dtype=next(target.parameters()).dtype).eval()


@torch.inference_mode()
def _vispec(
    target,
    processor,
    draft,
    prompts,
    max_new_tokens: int,
    device: torch.device,
    reference_outputs=None,
    fixed_output_length: bool = False,
    *,
    benchmark_name: str = "benchmark",
    proposal_mode: str = "tree",
) -> dict[str, Any]:
    """Run batch-one ViSpec/MSD rounds with cached target verification.

    Args:
        target: Target VLM used to verify each proposal.
        processor: Multimodal processor paired with the target.
        draft: ViSpec draft model used to construct proposals.
        prompts: Batch-one multimodal conversations.
        max_new_tokens: Maximum generated tokens per prompt.
        device: Device used for target and draft inference.
        reference_outputs: Optional target token IDs used for parity metrics.
        fixed_output_length: Whether every prompt emits exactly ``max_new_tokens``.
        benchmark_name: Benchmark name included in progress logs.
        proposal_mode: ``tree`` keeps ViSpec's multi-branch proposal; ``chain``
            retains only the top-1 token at each draft step.

    Returns:
        Aggregate throughput, acceptance, and optional parity metrics.
    """
    from nemo_automodel.components.speculative.eagle.vispec_decode import VispecCachedGreedyDecoder
    from nemo_automodel.components.speculative.eagle.vispec_target import HFVispecTargetModel

    image_token_id = int(getattr(target.config, "image_token_id"))
    vispec_target = HFVispecTargetModel(target, image_token_id=image_token_id)

    decoder = VispecCachedGreedyDecoder(vispec_target, draft)
    proposal_top_k = VISPEC_TOP_K if proposal_mode == "tree" else 1
    proposal_beam_width = VISPEC_TOTAL_TOKEN - 1 if proposal_mode == "tree" else VISPEC_DEPTH + 1
    output_tokens = 0
    draft_tokens = 0
    accepted_tokens = 0
    verify_steps = 0
    exact_matches = 0
    matching_tokens = 0
    compared_tokens = 0
    common_prefix_tokens = 0
    if prompts:
        warmup_inputs = _prepare_inputs(processor, prompts[0], device)
        decoder.prefill(warmup_inputs)
        decoder.decode_round(
            draft_steps=VISPEC_DEPTH,
            top_k=proposal_top_k,
            beam_width=proposal_beam_width,
        )
    _sync(device)
    start = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        logger.info(
            "[vispec] benchmark=%s sample=%d/%d started",
            benchmark_name,
            prompt_index + 1,
            len(prompts),
        )
        model_inputs = _prepare_inputs(processor, prompt, device)
        decoder.prefill(model_inputs)
        sample_max_new_tokens = (
            len(reference_outputs[prompt_index])
            if fixed_output_length and reference_outputs is not None
            else max_new_tokens
        )
        generated = 0
        generated_ids: list[int] = []
        while generated < sample_max_new_tokens:
            proposal, result = decoder.decode_round(
                draft_steps=VISPEC_DEPTH,
                top_k=proposal_top_k,
                beam_width=proposal_beam_width,
            )
            emitted = list(result.accepted_token_ids[: sample_max_new_tokens - generated])
            eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
            if not fixed_output_length and eos_token_id in emitted:
                emitted = emitted[: emitted.index(eos_token_id) + 1]
            if not emitted:
                if fixed_output_length:
                    raise RuntimeError("ViSpec emitted no token before reaching the fixed output length.")
                break
            generated_ids.extend(emitted)
            generated += len(emitted)
            output_tokens += len(emitted)
            draft_tokens += len(proposal.nodes)
            accepted_tokens += min(result.accepted_draft_tokens, max(0, len(emitted) - 1))
            verify_steps += 1
            if not fixed_output_length and eos_token_id in emitted:
                break
        _report_sample_progress(
            mode="vispec",
            benchmark_name=benchmark_name,
            sample_index=prompt_index + 1,
            total_samples=len(prompts),
            output_tokens=output_tokens,
            start_time=start,
        )
        if reference_outputs is not None:
            reference_ids = reference_outputs[prompt_index]
            exact_matches += int(generated_ids == reference_ids)
            compared_length = min(len(generated_ids), len(reference_ids))
            matching_tokens += sum(
                generated_ids[token_index] == reference_ids[token_index] for token_index in range(compared_length)
            )
            compared_tokens += max(len(generated_ids), len(reference_ids))
            for generated_id, reference_id in zip(generated_ids, reference_ids):
                if generated_id != reference_id:
                    break
                common_prefix_tokens += 1
    _sync(device)
    wall = time.perf_counter() - start
    accept_length, emitted_tokens_per_step = _acceptance_lengths(accepted_tokens, verify_steps)
    return {
        "completed": len(prompts),
        "output_tokens": output_tokens,
        "wall_clock_s": wall,
        "tok_s": output_tokens / wall,
        "accept_length": accept_length,
        "emitted_tokens_per_step": emitted_tokens_per_step,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        "exact_match_count": exact_matches if reference_outputs is not None else None,
        "exact_match_rate": exact_matches / len(prompts) if reference_outputs is not None and prompts else None,
        "token_match_rate": matching_tokens / compared_tokens
        if reference_outputs is not None and compared_tokens
        else None,
        "mean_common_prefix_length": common_prefix_tokens / len(prompts)
        if reference_outputs is not None and prompts
        else None,
    }


def main() -> None:
    """Run the requested batch-one speculative decoding benchmark."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--mode", choices=("baseline", "dflash", "vispec"), required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument(
        "--fixed-output-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate exactly max-new-tokens per sample (default: enabled).",
    )
    parser.add_argument("--block-size", type=int, choices=(4, 8, 16))
    parser.add_argument("--draft-layers", type=int, choices=(1, 3, 5))
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--verification-mode", choices=("block", "sequential"), default="block")
    parser.add_argument(
        "--visual-gate-multiplier",
        type=float,
        default=1.0,
        help="DFlash pooled-MLP ablation: multiply checkpoint visual gates after loading.",
    )
    parser.add_argument(
        "--draft-image-context-mode",
        choices=("keep", "zero", "shuffle"),
        default="keep",
        help="DFlash diagnostic: preserve, zero, or reorder image-token target features only on the draft side.",
    )
    parser.add_argument(
        "--vispec-proposal-mode",
        choices=("tree", "chain"),
        default="tree",
        help="Use the default multi-branch ViSpec tree or a top-1 chain ablation.",
    )
    parser.add_argument("--only", choices=REPRESENTATIVE_BENCHMARKS + LEGACY_BENCHMARKS + TEXT_BENCHMARKS)
    parser.add_argument(
        "--benchmark-suite",
        choices=("representative", "legacy", "text"),
        default="representative",
        help="Run the current four-dataset suite or the original five-dataset suite standardized to 256 tokens.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="Completed baseline JSON used to calculate speedup without rerunning autoregressive decoding.",
    )
    parser.add_argument(
        "--check-target-parity",
        action="store_true",
        help="Generate greedy target references in memory and compare every speculative output token.",
    )
    parser.add_argument("--input-data", type=Path, help="Local ShareGPT-style multimodal JSONL.")
    parser.add_argument("--media-dir", type=Path, help="Root for relative image paths in --input-data.")
    parser.add_argument("--input-start", type=int, default=0, help="First JSONL row used by the local benchmark.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=FIXED_NEW_TOKENS,
        help="Generation length for --input-data; official suites use their YAML value.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input_data is not None and args.media_dir is None:
        parser.error("--media-dir is required with --input-data")
    if args.input_data is None and args.media_dir is not None:
        parser.error("--media-dir requires --input-data")
    if args.input_data is not None and args.only is not None:
        parser.error("--only cannot be combined with --input-data")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.num_prompts <= 0:
        parser.error("--num-prompts must be positive")
    if not math.isfinite(args.visual_gate_multiplier) or args.visual_gate_multiplier < 0:
        parser.error("--visual-gate-multiplier must be finite and non-negative")
    if args.mode != "dflash" and args.visual_gate_multiplier != 1.0:
        parser.error("--visual-gate-multiplier is only valid in dflash mode")
    if args.mode != "dflash" and args.draft_image_context_mode != "keep":
        parser.error("--draft-image-context-mode is only valid in dflash mode")
    if args.benchmark_suite == "text" and args.draft_image_context_mode != "keep":
        parser.error("--draft-image-context-mode zero/shuffle requires a multimodal benchmark suite")
    if args.mode == "baseline" and args.baseline_results is not None:
        parser.error("--baseline-results is only valid for dflash and vispec modes")
    if args.check_target_parity and args.mode != "dflash":
        parser.error("--check-target-parity is only valid in dflash mode")

    if args.input_data is not None:
        benchmark_inputs = [
            (
                "local_jsonl",
                args.max_new_tokens,
                _load_sharegpt_vlm_prompts(
                    args.input_data,
                    args.media_dir,
                    start=args.input_start,
                    limit=args.num_prompts,
                ),
            )
        ]
    else:
        benchmark_inputs = []
        suites = {
            "representative": (BENCHMARK_CONFIG, REPRESENTATIVE_BENCHMARKS, FIXED_NEW_TOKENS),
            "legacy": (LEGACY_BENCHMARK_CONFIG, LEGACY_BENCHMARKS, LEGACY_NEW_TOKENS),
            "text": (TEXT_BENCHMARK_CONFIG, TEXT_BENCHMARKS, LEGACY_NEW_TOKENS),
        }
        benchmark_config, expected_benchmarks, expected_max_new_tokens = suites[args.benchmark_suite]
        if args.only is not None and args.only not in expected_benchmarks:
            parser.error(f"--only {args.only} is not part of the {args.benchmark_suite} benchmark suite")
        for spec in _load_benchmark_specs(benchmark_config, expected_benchmarks, expected_max_new_tokens):
            if args.only is not None and spec["name"] != args.only:
                continue
            logger.info("[setup] loading benchmark=%s prompts=%d", spec["name"], args.num_prompts)
            prompts = (
                _load_text_prompts(spec, args.num_prompts)
                if args.benchmark_suite == "text"
                else _load_official_prompts(spec, args.num_prompts)
            )
            benchmark_inputs.append((spec["name"], spec["max_new_tokens"], prompts))

    benchmark_limits = {name: max_new_tokens for name, max_new_tokens, _ in benchmark_inputs}
    baseline_throughputs = (
        _load_baseline_throughput(
            args.baseline_results,
            benchmark_limits,
            num_prompts=args.num_prompts,
            fixed_output_length=args.fixed_output_length,
        )
        if args.baseline_results is not None
        else {}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[setup] loading target=%s", args.target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = _load_target(args.target, device, args.attn_implementation)
    processor = AutoProcessor.from_pretrained(args.target)
    results: dict[str, Any] = {}
    for benchmark_index, (name, max_new_tokens, prompts) in enumerate(benchmark_inputs, start=1):
        logger.info(
            "[%s] starting benchmark=%s (%d/%d) samples=%d tokens_per_sample=%d",
            args.mode,
            name,
            benchmark_index,
            len(benchmark_inputs),
            len(prompts),
            max_new_tokens,
        )
        if args.mode == "baseline":
            result = _baseline(
                target,
                processor,
                prompts,
                max_new_tokens,
                device,
                fixed_output_length=args.fixed_output_length,
                benchmark_name=name,
            )
        elif args.mode == "dflash":
            draft = _load_dflash(
                args.draft,
                target,
                device,
                visual_gate_multiplier=args.visual_gate_multiplier,
            )
            if args.block_size is not None:
                draft.block_size = args.block_size
            if args.draft_layers is not None:
                draft.layers = draft.layers[: args.draft_layers]
            reference_outputs = None
            if args.check_target_parity:
                reference_result = _baseline(
                    target,
                    processor,
                    prompts,
                    max_new_tokens,
                    device,
                    fixed_output_length=args.fixed_output_length,
                    benchmark_name=name,
                )
                reference_outputs = reference_result["_reference_outputs"]
            result = _dflash(
                target,
                processor,
                draft,
                prompts,
                max_new_tokens,
                device,
                reference_outputs=reference_outputs,
                fixed_output_length=args.fixed_output_length,
                verification_mode=args.verification_mode,
                benchmark_name=name,
                draft_image_context_mode=args.draft_image_context_mode,
            )
        else:
            result = _vispec(
                target,
                processor,
                _load_vispec(args.draft, target, device),
                prompts,
                max_new_tokens,
                device,
                None,
                fixed_output_length=args.fixed_output_length,
                benchmark_name=name,
                proposal_mode=args.vispec_proposal_mode,
            )
        if args.mode == "baseline":
            result.pop("_reference_outputs", None)
        if name in baseline_throughputs:
            result["baseline_tok_s"] = baseline_throughputs[name]
            result["speedup_vs_cached_autoregressive"] = result["tok_s"] / baseline_throughputs[name]
        results[name] = {
            "target": args.target,
            "draft": args.draft,
            "mode": args.mode,
            "num_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "fixed_output_length": args.fixed_output_length,
            "benchmark_suite": args.benchmark_suite if args.input_data is None else None,
            "input_data": str(args.input_data) if args.input_data is not None else None,
            "media_dir": str(args.media_dir) if args.media_dir is not None else None,
            "input_start": args.input_start if args.input_data is not None else None,
            "baseline_results": str(args.baseline_results) if args.baseline_results is not None else None,
            "target_parity_checked": args.check_target_parity if args.mode == "dflash" else None,
            "block_size": args.block_size if args.mode == "dflash" else None,
            "draft_layers": args.draft_layers if args.mode == "dflash" else None,
            "visual_gate_multiplier": args.visual_gate_multiplier if args.mode == "dflash" else None,
            "draft_image_context_mode": args.draft_image_context_mode if args.mode == "dflash" else None,
            "attn_implementation": args.attn_implementation,
            "verification_mode": args.verification_mode
            if args.mode == "dflash"
            else (args.vispec_proposal_mode if args.mode == "vispec" else None),
            "vispec_depth": VISPEC_DEPTH if args.mode == "vispec" else None,
            "vispec_top_k": (VISPEC_TOP_K if args.vispec_proposal_mode == "tree" else 1)
            if args.mode == "vispec"
            else None,
            "vispec_total_token": (VISPEC_TOTAL_TOKEN if args.vispec_proposal_mode == "tree" else VISPEC_DEPTH + 2)
            if args.mode == "vispec"
            else None,
            **result,
        }
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        logger.info(
            "[%s] completed benchmark=%s (%d/%d) tok_s=%.2f; partial results saved to %s",
            args.mode,
            name,
            benchmark_index,
            len(benchmark_inputs),
            results[name]["tok_s"],
            args.output,
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
