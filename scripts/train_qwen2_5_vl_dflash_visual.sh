#!/usr/bin/env bash
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

set -euo pipefail

DFLASH_NPROC="${DFLASH_NPROC:-1}"
DFLASH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_CONFIG="examples/speculative/dflash/qwen2_5_vl_dflash_visual_stage1.yaml"
STAGE2_CONFIG="examples/speculative/dflash/qwen2_5_vl_dflash_visual_stage2.yaml"
STAGE1_CHECKPOINT_ROOT="dflash_visual_stage1_checkpoints"
cd "${DFLASH_REPO_ROOT}"

echo "Starting DFlash visual Stage 1 (visual adaptor, one epoch)."
uv run torchrun --standalone --nproc_per_node="${DFLASH_NPROC}" \
  -m nemo_automodel.recipes.llm.train_dflash \
  --config "${STAGE1_CONFIG}"

STAGE1_DRAFT="$(
  find "${STAGE1_CHECKPOINT_ROOT}" -type f -path '*/model/consolidated/config.json' -printf '%T@ %h\n' \
    | sort -nr \
    | sed -n '1{s/^[^ ]* //;p;}'
)"
if [[ -z "${STAGE1_DRAFT}" ]]; then
  echo "Stage 1 completed without producing a consolidated checkpoint." >&2
  exit 1
fi
if [[ ! -f "${STAGE1_DRAFT}/model.safetensors" && ! -f "${STAGE1_DRAFT}/model.safetensors.index.json" ]]; then
  echo "Stage 1 checkpoint ${STAGE1_DRAFT} has no readable safetensors manifest." >&2
  exit 1
fi

echo "Stage 1 completed. Starting Stage 2 from ${STAGE1_DRAFT}."
uv run torchrun --standalone --nproc_per_node="${DFLASH_NPROC}" \
  -m nemo_automodel.recipes.llm.train_dflash \
  --config "${STAGE2_CONFIG}" \
  --recipe_args.draft_init_from "${STAGE1_DRAFT}"

echo "DFlash visual Stage 1 and Stage 2 completed successfully."
