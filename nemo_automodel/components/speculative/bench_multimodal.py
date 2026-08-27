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

"""Multimodal dataset adapters for speculative-decoding HTTP benchmarks."""

from __future__ import annotations

import ast
import base64
import io
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable


class MultimodalBenchmark(str, Enum):
    """Supported benchmark row schemas."""

    SCIENCEQA = "scienceqa"
    MMVET = "mmvet"
    TEXTVQA = "textvqa"
    MME = "mme"
    COCO_CAPTION = "coco_caption"
    VIZWIZ = "vizwiz"
    GQA = "gqa"
    SEED_BENCH = "seed_bench"
    CHARXIV_REASONING = "charxiv_reasoning"
    MMMU_PRO = "mmmu_pro"


def _image_data_url(image: Any) -> str:
    """Convert an HF image value, local path, URL, or bytes into an image URL."""
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:image/")):
            return image
        path = Path(image)
        if path.is_file():
            suffix = path.suffix.lower().lstrip(".") or "jpeg"
            mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
            return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
        raise ValueError(f"Image string is neither a URL nor an existing file: {image!r}")

    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = image["bytes"]
        elif image.get("path"):
            return _image_data_url(image["path"])
        elif image.get("src"):
            return _image_data_url(image["src"])

    if isinstance(image, (bytes, bytearray)):
        return f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"

    if hasattr(image, "save"):
        output = io.BytesIO()
        image_format = str(getattr(image, "format", None) or "JPEG").upper()
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            image_format = "JPEG"
        if image_format == "JPEG" and getattr(image, "mode", "RGB") not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(output, format=image_format)
        mime = "jpeg" if image_format == "JPEG" else image_format.lower()
        return f"data:image/{mime};base64,{base64.b64encode(output.getvalue()).decode()}"

    raise ValueError(f"Unsupported image value of type {type(image).__name__}")


def _benchmark_prompt(text: str | None, image: Any, instruction: str) -> list[dict[str, Any]] | None:
    """Build one single-image benchmark conversation without a system message."""
    if not isinstance(text, str) or not text.strip() or image is None:
        return None
    user_content: list[dict[str, Any]] = [{"type": "text", "text": text.strip()}]
    if instruction:
        user_content.append({"type": "text", "text": instruction})
    user_content.append({"type": "image_url", "image_url": {"url": _image_data_url(image)}})
    return [{"role": "user", "content": user_content}]


def _multiple_choice_text(row: dict[str, Any]) -> str | None:
    """Format a ScienceQA or SEED-Bench row as an explained multiple choice task."""
    question = row.get("question")
    choices = row.get("choices")
    if choices is None:
        choices = [row.get(f"choice_{label.lower()}") for label in "ABCD"]
        choices = [choice for choice in choices if isinstance(choice, str) and choice]
    if not isinstance(question, str) or not isinstance(choices, list) or not choices:
        return None
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    options = " ".join(f"({label}) {choice}" for label, choice in zip(labels, choices))
    hint = row.get("hint")
    context = hint.strip() if isinstance(hint, str) and hint.strip() else "N/A"
    return (
        f"Context: {context}\nQuestion: {question}\nOptions: {options}\n"
        'Your answer should begin with "The answer is". Please answer with an explanation. Answer:'
    )


def _mmmu_pro_prompt(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Build one interleaved multi-image MMMU-Pro conversation."""
    question = row.get("question")
    options = row.get("options")
    if not isinstance(question, str) or not question.strip() or not isinstance(options, str):
        return None
    try:
        parsed_options = ast.literal_eval(options)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed_options, list) or not all(isinstance(option, str) for option in parsed_options):
        return None
    labels = [chr(ord("A") + index) for index in range(len(parsed_options))]
    option_text = " ".join(f"({label}) {option}" for label, option in zip(labels, parsed_options))
    text = f"{question.strip()}\nOptions: {option_text}"
    content: list[dict[str, Any]] = []
    for part in re.split(r"(<image\s+\d+>)", text):
        match = re.fullmatch(r"<image\s+(\d+)>", part)
        if match is None:
            if part.strip():
                content.append({"type": "text", "text": part.strip()})
            continue
        image = row.get(f"image_{match.group(1)}")
        if image is None:
            return None
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(image)}})
    if not any(item["type"] == "image_url" for item in content):
        return None
    content.append(
        {
            "type": "text",
            "text": 'Your answer should begin with "The answer is". Please answer with an explanation.',
        }
    )
    return [{"role": "user", "content": content}]


def adapt_multimodal_row(row: dict[str, Any], benchmark: MultimodalBenchmark) -> list[dict[str, Any]] | None:
    """Convert one supported benchmark row into OpenAI Vision messages."""
    if benchmark is MultimodalBenchmark.SCIENCEQA:
        return _benchmark_prompt(_multiple_choice_text(row), row.get("image"), "")
    if benchmark is MultimodalBenchmark.TEXTVQA:
        return _benchmark_prompt(
            row.get("question"),
            row.get("image"),
            "Perform an OCR task on the provided image. Please extract the text accurately and provide a detailed "
            "explanation of the process. Ensure the response is comprehensive and well-structured.",
        )
    if benchmark is MultimodalBenchmark.COCO_CAPTION:
        return _benchmark_prompt(
            "Please provide a detailed description of the given image.",
            row.get("image"),
            "",
        )
    if benchmark is MultimodalBenchmark.SEED_BENCH:
        images = row.get("image")
        if row.get("data_type") != "image" or not isinstance(images, list) or not images:
            return None
        return _benchmark_prompt(_multiple_choice_text(row), images[0], "")
    if benchmark is MultimodalBenchmark.CHARXIV_REASONING:
        return _benchmark_prompt(row.get("reasoning_q"), row.get("image"), "Please answer with an explanation.")
    if benchmark is MultimodalBenchmark.MMMU_PRO:
        return _mmmu_pro_prompt(row)
    if benchmark in {
        MultimodalBenchmark.MMVET,
        MultimodalBenchmark.MME,
        MultimodalBenchmark.VIZWIZ,
        MultimodalBenchmark.GQA,
    }:
        question = row.get("text") if benchmark is MultimodalBenchmark.GQA else row.get("question")
        return _benchmark_prompt(question, row.get("image"), "Please answer with an explanation.")
    raise ValueError(f"Unsupported multimodal benchmark: {benchmark}")


def load_multimodal_prompts(
    args: Any,
    load_rows: Callable[..., Iterable[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Load and adapt a multimodal benchmark, including GQA's split image table."""
    benchmark = MultimodalBenchmark(args.benchmark_adapter)
    rows = load_rows(
        args.input_data,
        split=args.split,
        name=args.dataset_name,
        shuffle_seed=args.shuffle_seed,
    )

    image_by_id: dict[str, Any] | None = None
    if benchmark is MultimodalBenchmark.GQA:
        if not args.dataset_name or not args.dataset_name.endswith("_instructions"):
            raise ValueError("GQA requires an *_instructions dataset_name so its matching image config can be joined.")
        instruction_rows = []
        needed_image_ids = set()
        for row in rows:
            instruction_rows.append(row)
            needed_image_ids.add(row.get("imageId"))
            if len(instruction_rows) >= args.num_prompts:
                break
        rows = instruction_rows
        image_rows = load_rows(
            args.input_data,
            split=args.split,
            name=args.dataset_name.removesuffix("_instructions") + "_images",
            shuffle_seed=None,
        )
        image_by_id = {}
        for row in image_rows:
            image_id = row.get("id")
            if image_id in needed_image_ids:
                image_by_id[image_id] = row.get("image")
                if len(image_by_id) == len(needed_image_ids):
                    break

    prompts: list[list[dict[str, Any]]] = []
    seen_coco_ids: set[Any] = set()
    for raw_row in rows:
        row = dict(raw_row)
        if image_by_id is not None:
            row["image"] = image_by_id.get(row.get("imageId"))
            row["text"] = row.get("question")
        if benchmark is MultimodalBenchmark.COCO_CAPTION:
            image_id = row.get("id")
            if image_id in seen_coco_ids:
                continue
            seen_coco_ids.add(image_id)
        prompt = adapt_multimodal_row(row, benchmark)
        if prompt is not None:
            prompts.append(prompt)
        if len(prompts) >= args.num_prompts:
            break
    return prompts
