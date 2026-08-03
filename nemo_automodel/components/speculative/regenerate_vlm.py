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

"""Regenerate image-conversation answers with a vision-language target model.

ViSpec's stage-2 corpus is built by throwing away the original (short) captions
of an image-caption dataset and having the *target VLM itself* answer each
prompt, with a length instruction appended so the answers are long enough to
train a draft on. Two properties matter:

* **On-policy.** The draft is supervised on the target's own output
  distribution, so the training text has to come from the target.
* **Long assistant turns.** Public multimodal SFT corpora answer in a sentence
  or two; a draft trained on those never sees the long-form decoding it has to
  accelerate. ViSpec appends "Please answer with at least 1000 words." to every
  prompt to force long generations.

The output is ShareGPT-style JSONL plus a meta JSON, which
``nemo_automodel.components.datasets.vlm.datasets.make_meta_dataset`` reads
directly, so the ViSpec recipe consumes it without a bespoke dataset class.

Example::

    uv run python -m nemo_automodel.components.speculative.regenerate_vlm \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --dataset liuhaotian/LLaVA-Pretrain \\
        --image-root /data/LLaVA-Pretrain \\
        --output-dir /data/vispec_stage2 \\
        --engine vllm --batch-size 8 \\
        --end 68000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from nemo_automodel.components.datasets.vlm.datasets import convert_sharegpt_to_conversation
from nemo_automodel.components.datasets.vlm.utils import set_image_pixel_bounds
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

HAVE_PIL, PIL_Image = safe_import("PIL.Image")
HAVE_VLLM, VLLM = safe_import("vllm")

LENGTH_INSTRUCTION = "Please answer with at least 1000 words."
IMAGE_PLACEHOLDER = "<image>"

# The sharegpt mapping this script writes into meta.json. The round-trip check
# parses with the same two dicts, so it cannot pass against a layout the
# generated meta.json no longer declares.
SHAREGPT_COLUMNS = {"messages": "conversations", "images": "images"}
SHAREGPT_TAGS = {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}


@dataclass
class RegenerationConfig:
    """Settings for one regeneration run.

    Attributes:
        model: Target VLM id or path; it both prompts and answers.
        dataset: HuggingFace dataset id or local path of the source corpus.
        split: Split expression passed to ``load_dataset``.
        image_root: Directory the dataset's relative image paths resolve against.
        output_dir: Destination directory for ``data.jsonl`` and ``meta.json``.
        start: First source-row index to regenerate (after shuffling).
        end: One past the last source-row index to regenerate.
        max_new_tokens: Generation cap per answer.
        temperature: Sampling temperature; ``0`` means greedy.
        shuffle_seed: Seed for the source-corpus shuffle, so ``start``/``end``
            slices are reproducible across shards.
        length_instruction: Sentence appended to every prompt to force a long
            answer. Empty disables the append.
        image_max_pixels: Per-image pixel ceiling handed to the processor, or
            ``None`` to keep its default. Set it to the same value stage-2
            training uses: the answer is conditioned on whatever resolution the
            target saw here, so regenerating at a higher one supervises the
            draft against an image it never gets at training time.
        image_min_pixels: Per-image pixel floor, or ``None`` for the default.
        engine: Inference engine used to regenerate answers.
        batch_size: Number of requests submitted to vLLM at once. The Hugging
            Face engine processes one request at a time.
        tensor_parallel_size: Number of GPUs used by the vLLM model replica.
        gpu_memory_utilization: Fraction of each GPU reserved by vLLM.
        max_model_len: Maximum prompt plus generation length used by vLLM.
        trust_remote_code: Whether model repositories may execute custom code.
    """

    model: str
    dataset: str
    split: str
    image_root: str
    output_dir: str
    start: int
    end: int
    max_new_tokens: int
    temperature: float
    shuffle_seed: int
    length_instruction: str
    image_max_pixels: int | None = None
    image_min_pixels: int | None = None
    engine: Literal["hf", "vllm"] = "hf"
    batch_size: int = 8
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    trust_remote_code: bool = False

    def build(self, *, processor: AutoProcessor) -> _AnswerGenerator:
        """Build the configured answer generator around an initialized processor.

        Args:
            processor: Processor loaded from the same target checkpoint.

        Returns:
            The configured Hugging Face or vLLM answer generator.
        """
        if self.engine == "hf":
            return _HfAnswerGenerator(_load_target_model(self.model), processor, self)
        if self.engine == "vllm":
            return _VllmAnswerGenerator(processor, self)
        raise ValueError(f"Unsupported regeneration engine {self.engine!r}; expected 'hf' or 'vllm'.")


@dataclass
class _PreparedSample:
    """One validated source row ready for target generation."""

    prompt_text: str
    image_path: str
    absolute_image: str
    content: list[dict]


class _AnswerGenerator(Protocol):
    """Backend contract for batched answer generation."""

    def generate(self, samples: list[_PreparedSample]) -> list[str]:
        """Generate one answer per sample, preserving input order."""
        ...


def _extract_prompt(example: dict) -> tuple[str, str] | None:
    """Pull the first human turn and image path out of a ShareGPT-style row.

    Args:
        example: One source row, with a ``conversations`` list and an ``image`` path.

    Returns:
        ``(prompt_text, image_path)``, or ``None`` when the row has no usable
        human turn or no image.
    """
    image_path = example.get("image")
    conversations = example.get("conversations") or []
    if not image_path:
        return None
    for turn in conversations:
        if turn.get("from") != "human":
            continue
        text = turn.get("value", "").replace(IMAGE_PLACEHOLDER, "").strip()
        return text, image_path
    return None


def _build_user_content(prompt_text: str, image_path: str, length_instruction: str) -> list[dict]:
    """Build the user-turn content parts, in the order the target sees them.

    The order (source text, image, length instruction) matches the reference
    implementation's ``preprocess_function``.

    Args:
        prompt_text: The source row's human turn, image placeholder stripped.
        image_path: Path to the image for this row.
        length_instruction: Sentence appended after the image, or empty.

    Returns:
        The content list of a single multimodal user turn.
    """
    content: list[dict] = []
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})
    content.append({"type": "image", "image": image_path})
    if length_instruction:
        content.append({"type": "text", "text": length_instruction})
    return content


def _serialize_human_turn(prompt_text: str, length_instruction: str) -> str:
    """Serialize the generation prompt back into one ShareGPT ``value`` string.

    ``make_meta_dataset`` rebuilds the user turn by splitting this string on the
    ``<image>`` placeholder *positionally*, so writing the parts in the order
    they were generated is what makes the training prefix reproduce the
    generation prefix. :func:`_assert_prompt_round_trip` verifies that per row.

    Args:
        prompt_text: The source row's human turn, image placeholder stripped.
        length_instruction: Sentence appended after the image, or empty.

    Returns:
        The ``value`` written to the human turn of ``data.jsonl``.
    """
    parts = [part for part in (prompt_text, IMAGE_PLACEHOLDER, length_instruction) if part]
    return "\n".join(parts)


def _assert_prompt_round_trip(row: dict, expected_content: list[dict], image_root: str) -> None:
    """Fail if the serialized row does not rebuild the exact generation prompt.

    The answer is only a valid supervision target for the prompt it was sampled
    under, so the prompt stage 2 reconstructs has to be the prompt this script
    generated from. That reconstruction happens in
    :func:`convert_sharegpt_to_conversation`, which is what
    ``make_meta_dataset`` runs over ``data.jsonl``, so the check calls the real
    parser rather than a copy of its rules: a change to either side that breaks
    the correspondence surfaces here instead of silently training the draft on a
    prompt the target never answered.

    Args:
        row: The row about to be written to ``data.jsonl``.
        expected_content: The user content that was handed to the target.
        image_root: The ``media_dir`` recorded in ``meta.json``.

    Raises:
        ValueError: If the rebuilt user turn differs from ``expected_content``.
    """
    rebuilt = convert_sharegpt_to_conversation(row, columns=SHAREGPT_COLUMNS, tags=SHAREGPT_TAGS, media_dir=image_root)
    actual = rebuilt["conversation"][0]["content"]
    if actual != expected_content:
        raise ValueError(
            "The serialized prompt does not rebuild the prompt the answer was generated from.\n"
            f"generated: {expected_content}\nrebuilt:   {actual}"
        )


@torch.no_grad()
def _generate_answer(model, processor, messages: list[dict], config: RegenerationConfig) -> str:
    """Run the target once and decode its answer.

    Args:
        model: The loaded target VLM.
        processor: The target's processor.
        messages: Chat messages wrapping :func:`_build_user_content`.
        config: The active regeneration settings.

    Returns:
        The decoded assistant answer, without the prompt prefix.
    """
    inputs = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.temperature > 0,
        temperature=config.temperature if config.temperature > 0 else None,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def _write_meta(output_dir: Path, image_root: str) -> Path:
    """Write the meta JSON that ``make_meta_dataset`` reads.

    Args:
        output_dir: Directory holding the generated ``data.jsonl``.
        image_root: Directory the JSONL's relative image paths resolve against.

    Returns:
        Path to the written meta file.
    """
    meta_path = output_dir / "meta.json"
    meta = {
        "vispec_stage2": {
            "file_name": "data.jsonl",
            "columns": SHAREGPT_COLUMNS,
            "tags": SHAREGPT_TAGS,
            "media_dir": image_root,
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def _load_target_model(model_path: str) -> torch.nn.Module:
    """Load the target VLM and place it on the available local device.

    ``device_map="auto"`` requires the optional ``accelerate`` package. The
    regeneration utility only needs one local target replica, so explicit
    placement works in both minimal and Accelerate-enabled environments.

    Args:
        model_path: HuggingFace model identifier or local checkpoint directory.

    Returns:
        The evaluated target model on CUDA when available, otherwise CPU.
    """
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto").eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class _HfAnswerGenerator:
    """Sequential Hugging Face generation backend."""

    def __init__(self, model, processor, config: RegenerationConfig):
        self._model = model
        self._processor = processor
        self._config = config

    def generate(self, samples: list[_PreparedSample]) -> list[str]:
        """Generate one answer per sample, preserving input order."""
        return [
            _generate_answer(
                self._model,
                self._processor,
                [{"role": "user", "content": sample.content}],
                self._config,
            )
            for sample in samples
        ]


class _VllmAnswerGenerator:
    """Batched offline-vLLM generation backend for one-image requests."""

    def __init__(self, processor, config: RegenerationConfig):
        if not HAVE_VLLM:
            raise ImportError(
                "vLLM is required for --engine vllm. Install it with "
                "`uv pip install vllm==0.23.0 --torch-backend=auto`."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The vLLM regeneration engine requires an available GPU and NVIDIA driver.")
        mm_processor_kwargs = {
            key: value
            for key, value in {
                "max_pixels": config.image_max_pixels,
                "min_pixels": config.image_min_pixels,
            }.items()
            if value is not None
        }
        engine_kwargs = {
            "model": config.model,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "max_num_seqs": config.batch_size,
            "limit_mm_per_prompt": {"image": 1},
            "trust_remote_code": config.trust_remote_code,
            "seed": config.shuffle_seed,
        }
        if mm_processor_kwargs:
            engine_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        self._llm = VLLM.LLM(**engine_kwargs)
        self._sampling_params = VLLM.SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )
        self._processor = processor

    def generate(self, samples: list[_PreparedSample]) -> list[str]:
        """Generate one answer per sample, preserving input order."""
        requests = []
        images = []
        try:
            for sample in samples:
                messages = [{"role": "user", "content": sample.content}]
                prompt = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image = PIL_Image.open(sample.absolute_image).convert("RGB")
                images.append(image)
                requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            outputs = self._llm.generate(requests, sampling_params=self._sampling_params, use_tqdm=False)
            return [output.outputs[0].text.strip() for output in outputs]
        finally:
            for image in images:
                image.close()


def _load_source_dataset(dataset_path: str, split: str):
    """Load a source corpus from either a dataset ID or a local JSON file.

    Args:
        dataset_path: HuggingFace dataset identifier or local JSON/JSONL file.
        split: Dataset split name passed to ``datasets.load_dataset``.

    Returns:
        The loaded HuggingFace dataset split.
    """
    if Path(dataset_path).is_file():
        return load_dataset("json", data_files=dataset_path, split=split)
    return load_dataset(dataset_path, split=split)


def regenerate(config: RegenerationConfig) -> Path:
    """Regenerate answers for a slice of the source corpus and write the shard.

    Args:
        config: The regeneration settings.

    Returns:
        Path to the written ``data.jsonl``.
    """
    if not HAVE_PIL:
        raise ImportError("Pillow is required to load images for VLM regeneration (`uv pip install pillow`).")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.jsonl"

    dataset = _load_source_dataset(config.dataset, config.split).shuffle(seed=config.shuffle_seed)
    dataset = dataset.select(range(config.start, min(config.end, len(dataset))))

    processor = AutoProcessor.from_pretrained(config.model)
    set_image_pixel_bounds(processor, max_pixels=config.image_max_pixels, min_pixels=config.image_min_pixels)
    generator = config.build(processor=processor)

    written = 0
    skipped = 0
    total = len(dataset)
    # One target forward per row over a corpus sized for draft training runs for
    # hours. Reporting on a cadence, and flushing as we go, is what makes a
    # stalled or empty run distinguishable from a slow one while it is still
    # running rather than only from the final count.
    log_every = max(1, total // 100)

    def write_batch(handle, samples: list[_PreparedSample]) -> tuple[int, int]:
        """Generate and write a prepared batch, returning written and skipped counts."""
        answers = generator.generate(samples)
        if len(answers) != len(samples):
            raise RuntimeError(
                f"The {config.engine} engine returned {len(answers)} answers for {len(samples)} requests."
            )
        batch_written = 0
        batch_skipped = 0
        for sample, answer in zip(samples, answers, strict=True):
            if not answer:
                batch_skipped += 1
                continue
            # The stored prompt is the generation prompt, length instruction and
            # part order included. Dropping the instruction would be closer to
            # the deployment prompt shape, but it would also supervise the draft
            # with an answer the target produced under a different prompt, and
            # the reference stores the features of the exact modified prompt.
            row = {
                "conversations": [
                    {
                        "from": "human",
                        "value": _serialize_human_turn(sample.prompt_text, config.length_instruction),
                    },
                    {"from": "gpt", "value": answer},
                ],
                "images": [sample.image_path],
            }
            _assert_prompt_round_trip(row, sample.content, config.image_root)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            batch_written += 1
        return batch_written, batch_skipped

    pending: list[_PreparedSample] = []
    effective_batch_size = config.batch_size if config.engine == "vllm" else 1
    with data_path.open("w") as handle:
        for index, example in enumerate(dataset):
            if index and index % log_every == 0:
                handle.flush()
                logger.info("Regenerated %d/%d rows (%d written, %d skipped)", index, total, written, skipped)
            extracted = _extract_prompt(example)
            if extracted is None:
                skipped += 1
                continue
            prompt_text, image_path = extracted
            absolute_image = os.path.join(config.image_root, image_path)
            if not os.path.exists(absolute_image):
                skipped += 1
                continue
            content = _build_user_content(prompt_text, absolute_image, config.length_instruction)
            pending.append(
                _PreparedSample(
                    prompt_text=prompt_text,
                    image_path=image_path,
                    absolute_image=absolute_image,
                    content=content,
                )
            )
            if len(pending) == effective_batch_size:
                batch_written, batch_skipped = write_batch(handle, pending)
                written += batch_written
                skipped += batch_skipped
                pending.clear()
        if pending:
            batch_written, batch_skipped = write_batch(handle, pending)
            written += batch_written
            skipped += batch_skipped

    meta_path = _write_meta(output_dir, config.image_root)
    logger.info("Wrote %d regenerated samples (%d skipped) to %s; meta at %s", written, skipped, data_path, meta_path)
    return data_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Regenerate VLM answers for ViSpec stage-2 training.")
    parser.add_argument("--model", required=True, help="Target VLM id or path.")
    parser.add_argument("--dataset", required=True, help="Source dataset id or path.")
    parser.add_argument("--split", default="train", help="Source split expression.")
    parser.add_argument("--image-root", required=True, help="Directory the dataset's image paths resolve against.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for data.jsonl and meta.json.")
    parser.add_argument("--start", type=int, default=0, help="First source-row index (after shuffling).")
    parser.add_argument("--end", type=int, default=68000, help="One past the last source-row index.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation cap per answer.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; 0 means greedy.")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed for the source-corpus shuffle.")
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=None,
        help="Per-image pixel ceiling; match recipe_args.image_max_pixels of the stage-2 config.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=None,
        help="Per-image pixel floor; match recipe_args.image_min_pixels of the stage-2 config.",
    )
    parser.add_argument(
        "--length-instruction",
        default=LENGTH_INSTRUCTION,
        help="Sentence appended to every prompt to force a long answer; empty disables it.",
    )
    parser.add_argument(
        "--engine",
        choices=("hf", "vllm"),
        default="hf",
        help="Generation engine. vLLM batches requests and requires a GPU.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Requests submitted to vLLM per batch.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used by the vLLM model replica.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU reserved by vLLM.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096, help="vLLM prompt plus generation limit.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow model repository custom code.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parses ``argv`` and returns the process exit code."""
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    if args.end <= args.start:
        raise ValueError(f"--end ({args.end}) must be greater than --start ({args.start})")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.tensor_parallel_size <= 0:
        raise ValueError(f"--tensor-parallel-size must be positive, got {args.tensor_parallel_size}")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError(f"--gpu-memory-utilization must be in (0, 1], got {args.gpu_memory_utilization}")
    if args.max_model_len <= 0:
        raise ValueError(f"--max-model-len must be positive, got {args.max_model_len}")
    regenerate(
        RegenerationConfig(
            model=args.model,
            dataset=args.dataset,
            split=args.split,
            image_root=args.image_root,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            shuffle_seed=args.shuffle_seed,
            length_instruction=args.length_instruction,
            image_max_pixels=args.image_max_pixels,
            image_min_pixels=args.image_min_pixels,
            engine=args.engine,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
