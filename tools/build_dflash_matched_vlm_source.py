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

"""Build a task-balanced image/prompt source for target VLM regeneration."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_TEMPLATES = (
    "<image>\nIdentify the main subject and state one clearly visible detail. Answer in at most 20 words.",
    "<image>\nTranscribe the most prominent readable text. If none is visible, answer 'No readable text.' "
    "Use at most 30 words.",
    "<image>\nDescribe the image in one or two concise sentences using at most 80 words.",
    "<image>\nWhat is most likely happening, and which visible evidence supports your conclusion? "
    "Answer in two to four sentences using at most 160 words.",
    "<image>\nGive a focused, detailed description of the image using at most 200 words.",
)


def _build_source(input_path: Path, output_path: Path, *, num_images: int, seed: int) -> int:
    """Write five task prompts for each deterministically selected image.

    Args:
        input_path: ShareGPT JSONL containing an ``images`` list per row.
        output_path: Destination JSONL accepted by ``regenerate_vlm``.
        num_images: Number of unique image paths to select.
        seed: Seed used to shuffle unique image paths before selection.

    Returns:
        Number of prompt rows written.
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}.")
    image_paths: list[str] = []
    seen: set[str] = set()
    with input_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}.") from error
            images = row.get("images")
            if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                continue
            if images[0] not in seen:
                seen.add(images[0])
                image_paths.append(images[0])
    if len(image_paths) < num_images:
        raise ValueError(f"Requested {num_images} unique images, but {input_path} contains {len(image_paths)}.")

    random.Random(seed).shuffle(image_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w") as handle:
        for image_path in image_paths[:num_images]:
            for prompt in PROMPT_TEMPLATES:
                row = {
                    "image": image_path,
                    "conversations": [{"from": "human", "value": prompt}],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return written


def main() -> None:
    """Build a deterministic matched-task regeneration source."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing ShareGPT VLM JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Destination source JSONL.")
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    written = _build_source(args.input, args.output, num_images=args.num_images, seed=args.seed)
    logger.info("Wrote %d prompts from %d images to %s", written, args.num_images, args.output)


if __name__ == "__main__":
    main()
