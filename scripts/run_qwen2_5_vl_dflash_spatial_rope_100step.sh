#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mkdir -p dflash_spatial_rope_100step benchmark_results

.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m nemo_automodel.recipes.llm.train_dflash \
  --config examples/speculative/dflash/qwen2_5_vl_dflash_spatial_rope_100step.yaml \
  2>&1 | tee dflash_spatial_rope_100step/train.log

checkpoint="dflash_spatial_rope_100step_checkpoints/epoch_2_step_100/model/consolidated"
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
  --output benchmark_results/dflash_spatial_rope_step100_legacy5_10x256.json \
  2>&1 | tee dflash_spatial_rope_100step/eval.log
