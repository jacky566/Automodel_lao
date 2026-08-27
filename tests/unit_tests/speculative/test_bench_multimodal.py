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

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_automodel.components.speculative.bench_multimodal import (
    MultimodalBenchmark,
    adapt_multimodal_row,
    load_multimodal_prompts,
)


@pytest.mark.parametrize(
    "benchmark",
    [
        MultimodalBenchmark.MMVET,
        MultimodalBenchmark.MME,
        MultimodalBenchmark.VIZWIZ,
    ],
)
def test_explanation_adapters_build_user_only_messages(benchmark):
    row = {"question": "What is shown?", "image": b"jpeg"}
    prompt = adapt_multimodal_row(row, benchmark)
    assert prompt is not None
    assert len(prompt) == 1
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"][0] == {"type": "text", "text": "What is shown?"}
    assert prompt[0]["content"][1] == {"type": "text", "text": "Please answer with an explanation."}
    assert prompt[0]["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_scienceqa_formats_context_and_choices():
    row = {
        "question": "Which animal is shown?",
        "choices": ["cat", "dog"],
        "hint": "It barks.",
        "image": b"jpeg",
    }
    prompt = adapt_multimodal_row(row, MultimodalBenchmark.SCIENCEQA)
    assert prompt is not None
    text = prompt[0]["content"][0]["text"]
    assert "Context: It barks." in text
    assert "Options: (A) cat (B) dog" in text
    assert 'begin with "The answer is"' in text


def test_textvqa_uses_official_ocr_instruction():
    prompt = adapt_multimodal_row(
        {"question": "Read the sign.", "image": b"jpeg"},
        MultimodalBenchmark.TEXTVQA,
    )
    assert prompt is not None
    assert "Perform an OCR task" in prompt[0]["content"][1]["text"]


def test_coco_caption_uses_official_description_prompt():
    prompt = adapt_multimodal_row({"image": b"jpeg"}, MultimodalBenchmark.COCO_CAPTION)
    assert prompt is not None
    assert prompt[0]["content"][0]["text"] == "Please provide a detailed description of the given image."


def test_seed_bench_formats_image_multiple_choice_rows_only():
    row = {
        "question": "What color is the car?",
        "choice_a": "red",
        "choice_b": "blue",
        "choice_c": "green",
        "choice_d": "black",
        "data_type": "image",
        "image": [b"jpeg"],
    }
    prompt = adapt_multimodal_row(row, MultimodalBenchmark.SEED_BENCH)
    assert prompt is not None
    assert "(A) red (B) blue (C) green (D) black" in prompt[0]["content"][0]["text"]
    row["data_type"] = "video"
    assert adapt_multimodal_row(row, MultimodalBenchmark.SEED_BENCH) is None


def test_gqa_loader_joins_instruction_and_image_configs():
    calls = []

    def load_rows(input_data, *, split, name, shuffle_seed):
        calls.append((input_data, split, name, shuffle_seed))
        if name.endswith("_images"):
            return [{"id": "image-a", "image": b"jpeg"}]
        return [{"imageId": "image-a", "question": "Is it overcast?"}]

    args = SimpleNamespace(
        benchmark_adapter="gqa",
        input_data="lmms-lab/GQA",
        split="testdev",
        dataset_name="testdev_balanced_instructions",
        shuffle_seed=7,
        num_prompts=4,
    )
    prompts = load_multimodal_prompts(args, load_rows)
    assert len(prompts) == 1
    assert calls == [
        ("lmms-lab/GQA", "testdev", "testdev_balanced_instructions", 7),
        ("lmms-lab/GQA", "testdev", "testdev_balanced_images", None),
    ]
    assert prompts[0][0]["content"][0]["text"] == "Is it overcast?"


def test_coco_loader_deduplicates_captions_for_the_same_image():
    args = SimpleNamespace(
        benchmark_adapter="coco_caption",
        input_data="lmms-lab/COCO-Caption2017",
        split="val",
        dataset_name=None,
        shuffle_seed=None,
        num_prompts=2,
    )
    rows = [
        {"id": 1, "image": b"first"},
        {"id": 1, "image": b"first"},
        {"id": 2, "image": b"second"},
    ]
    assert len(load_multimodal_prompts(args, lambda *args, **kwargs: rows)) == 2


def test_gqa_loader_requires_instruction_config():
    args = SimpleNamespace(
        benchmark_adapter="gqa",
        input_data="lmms-lab/GQA",
        split="testdev",
        dataset_name="testdev_balanced_images",
        shuffle_seed=None,
        num_prompts=1,
    )
    with pytest.raises(ValueError, match="instructions"):
        load_multimodal_prompts(args, lambda *args, **kwargs: [])


def test_charxiv_reasoning_uses_reasoning_question_and_image():
    prompt = adapt_multimodal_row(
        {"reasoning_q": "Which line declines fastest?", "image": b"jpeg"},
        MultimodalBenchmark.CHARXIV_REASONING,
    )

    assert prompt is not None
    assert prompt[0]["content"][0]["text"] == "Which line declines fastest?"
    assert prompt[0]["content"][-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_mmmu_pro_interleaves_numbered_images_and_options():
    prompt = adapt_multimodal_row(
        {
            "question": "Compare <image 1> with <image 2>.",
            "options": "['first', 'second', 'same', 'unknown']",
            "image_1": b"first",
            "image_2": b"second",
        },
        MultimodalBenchmark.MMMU_PRO,
    )

    assert prompt is not None
    content = prompt[0]["content"]
    assert [item["type"] for item in content] == ["text", "image_url", "text", "image_url", "text", "text"]
    assert "Options: (A) first (B) second (C) same (D) unknown" in content[-2]["text"]
