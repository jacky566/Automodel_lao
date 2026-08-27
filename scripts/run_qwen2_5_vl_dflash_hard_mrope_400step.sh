#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mkdir -p dflash_hard_mrope_400step benchmark_results

.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m nemo_automodel.recipes.llm.train_dflash \
  --config examples/speculative/dflash/qwen2_5_vl_dflash_hard_mrope_400step.yaml \
  2>&1 | tee dflash_hard_mrope_400step/train.log

checkpoint="dflash_hard_mrope_400step_checkpoints/epoch_8_step_400/model/consolidated"
if [[ ! -s "${checkpoint}/model.safetensors.index.json" ]]; then
  echo "Expected consolidated checkpoint was not created: ${checkpoint}" >&2
  exit 1
fi

.venv/bin/python tools/transformers_vlm_spec_bench.py \
  --target /data/models/Qwen2.5-VL-7B-Instruct \
  --draft "${checkpoint}" \
  --mode dflash \
  --benchmark-suite legacy \
  --num-prompts 10 \
  --fixed-output-length \
  --attn-implementation sdpa \
  --baseline-results benchmark_results/baseline_legacy5_10x256.json \
  --output benchmark_results/dflash_hard_mrope_step400_legacy5_10x256.json \
  2>&1 | tee dflash_hard_mrope_400step/eval.log
