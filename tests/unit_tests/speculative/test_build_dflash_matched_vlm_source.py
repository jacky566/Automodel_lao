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

"""Tests for the matched-task VLM regeneration source builder."""

import json

import pytest

from tools.build_dflash_matched_vlm_source import PROMPT_TEMPLATES, _build_source


def test_build_source_writes_balanced_prompts_without_answers(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({"images": [f"images/{index}.jpg"], "conversations": []}) for index in range(3)) + "\n"
    )

    written = _build_source(input_path, output_path, num_images=2, seed=7)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert written == 2 * len(PROMPT_TEMPLATES)
    assert len(rows) == written
    assert len({row["image"] for row in rows}) == 2
    assert all("gpt" not in {turn["from"] for turn in row["conversations"]} for row in rows)
    for image_path in {row["image"] for row in rows}:
        prompts = [row["conversations"][0]["value"] for row in rows if row["image"] == image_path]
        assert prompts == list(PROMPT_TEMPLATES)


def test_build_source_rejects_more_images_than_available(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({"images": ["one.jpg"]}) + "\n")

    with pytest.raises(ValueError, match="Requested 2 unique images"):
        _build_source(input_path, tmp_path / "output.jsonl", num_images=2, seed=1)
