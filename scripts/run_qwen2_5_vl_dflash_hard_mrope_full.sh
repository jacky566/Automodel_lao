#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

target_dir="/data/models/Qwen2.5-VL-7B-Instruct"
dataset_meta="/data/datasets/vispec_qwen2_5_vl/stage2/meta.json"
run_dir="/data/runs/dflash_qwen2_5_vl_hard_mrope_full"

if [[ ! -d "${target_dir}" ]]; then
  echo "Target model directory does not exist: ${target_dir}" >&2
  exit 1
fi
if [[ ! -f "${dataset_meta}" ]]; then
  echo "Dataset metadata does not exist: ${dataset_meta}" >&2
  exit 1
fi

mkdir -p "${run_dir}"

.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m nemo_automodel.recipes.llm.train_dflash \
  --config examples/speculative/dflash/qwen2_5_vl_dflash_hard_mrope_full.yaml \
  2>&1 | tee "${run_dir}/train.log"

checkpoint="${run_dir}/checkpoints/epoch_6_step_14574/model/consolidated"
if [[ ! -s "${checkpoint}/model.safetensors.index.json" ]]; then
  echo "Expected final consolidated checkpoint was not created: ${checkpoint}" >&2
  exit 1
fi

echo "Training complete: ${checkpoint}"
