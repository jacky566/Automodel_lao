# Qwen2.5-VL Hard-MRoPE Reproduction Agent Runbook

This file is an operational handoff for an agent reproducing the completed
Qwen2.5-VL DFlash Hard-MRoPE training run on a new server. It complements, and
does not replace, the repository-wide rules in `AGENTS.md`.

## Objective

Configure a clean NeMo AutoModel environment, download the exact source and
training inputs, validate them, and reproduce the full randomly initialized
Qwen2.5-VL Hard-MRoPE draft training run.

The reference run used:

- Source repository: `https://github.com/jacky566/Automodel_lao.git`
- Reproduction code commit: `d21fb5aa63b440333d0bc7305f9584daf9027d1c`
- Training annotations: `aaa23123/Vispec-stage2`
- Images: `liuhaotian/LLaVA-Pretrain`, file `images.zip`
- Target: `Qwen/Qwen2.5-VL-7B-Instruct`
- Config: `examples/speculative/dflash/qwen2_5_vl_dflash_hard_mrope_full.yaml`
- Samples: 68,000
- Epochs: 6
- Expected optimizer steps: 14,574
- Reference hardware: one NVIDIA B200
- Reference runtime: approximately 9 hours 46 minutes

The Hugging Face dataset `aaa23123/Vispec-stage2` contains only `meta.json`
and `data.jsonl`. It does not contain image files. Training is impossible until
the referenced images are present under the `media_dir` recorded in `meta.json`.

## Non-negotiable safety rules

- Read `AGENTS.md` and the repository skills it requires before changing code.
- Never print, save, commit, or paste an HF token, GitHub token, or other secret.
- Never upload datasets, images, model weights, checkpoints, optimizer state,
  logs, caches, or generated run directories to GitHub.
- Never use `git add .` or another broad staging command. Stage explicit files.
- Before any GitHub push, verify that the push URL is exactly
  `https://github.com/jacky566/Automodel_lao.git`.
- Push only to `origin/main`. Do not create a PR or push to NVIDIA or another
  person's repository unless the user explicitly changes the destination.
- Do not delete the original data or image archive without explicit approval.
- Do not silently change batch size, sequence length, epochs, learning rate,
  block size, anchors, MRoPE sections, or source commit. A changed value is a
  new experiment, not an exact reproduction.
- Do not start unrelated training, servers, `screen`, `tmux`, or background
  processes. Run the requested training in the foreground unless the user asks
  for a particular process manager.

## Success criteria

The run is complete only when all of the following are true:

1. The source commit is the pinned reproduction commit above.
2. `data.jsonl` contains exactly 68,000 rows.
3. Every image path referenced by `data.jsonl` exists below `media_dir`.
4. The target model has a readable `config.json` and model weights.
5. Training reaches epoch 6 and optimizer step 14,574 without non-finite loss.
6. The final consolidated safetensors checkpoint exists at
   `epoch_6_step_14574/model/consolidated` and contains no NaN or Inf tensors.
7. The training log and final checkpoint location are reported to the user.

## 1. Preflight the new server

Run these read-only checks first and report any failure before downloading:

```bash
set -euo pipefail

command -v git
command -v uv
command -v jq
command -v unzip
command -v nvidia-smi

df -h .
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
```

Reserve at least 110 GB of free disk space. The image archive is about 26 GB,
the extracted image files are about 28 GB, and keeping both temporarily uses
about 54 GB. The target model, uv environment, caches, logs, and training
checkpoints require additional space. If the server does not have enough
space, stop and ask the user to choose another storage location.

## 2. Clone and pin the source

If the repository is not present:

```bash
git clone https://github.com/jacky566/Automodel_lao.git
cd Automodel_lao
```

If it is already present, enter it without deleting or resetting user changes.
Check status before switching commits:

```bash
git status --short --branch
git remote -v
```

For a clean clone, pin the exact training implementation:

```bash
git switch --detach d21fb5aa63b440333d0bc7305f9584daf9027d1c
git rev-parse HEAD
```

Do not switch or reset a dirty checkout. If the checkout contains user changes,
stop and report them instead of overwriting them.

## 3. Build the environment with uv

Use the lockfile and the VLM media extras. Do not use `pip install`:

```bash
uv sync --frozen --extra vlm --extra vlm-media
uv run python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
```

If Hugging Face authentication is required, ask the user to authenticate
interactively. Never ask them to send the token in chat:

```bash
uv run hf auth login
```

## 4. Define reproducible local paths

Keep downloaded assets and runs outside the tracked source paths:

```bash
REPO_ROOT="$(pwd)"
REPRO_ROOT="${REPO_ROOT}/repro_assets"
DATASET_DIR="${REPRO_ROOT}/vispec_stage2"
MEDIA_DIR="${REPRO_ROOT}/LLaVA-Pretrain"
TARGET_DIR="${REPRO_ROOT}/models/Qwen2.5-VL-7B-Instruct"
RUN_DIR="${REPO_ROOT}/repro_runs/dflash_qwen2_5_vl_hard_mrope_full"

mkdir -p "${DATASET_DIR}" "${MEDIA_DIR}" "${TARGET_DIR}" "${RUN_DIR}"
```

Do not add `repro_assets/` or `repro_runs/` to Git.

## 5. Download annotations and images

Download the exact generated Stage-2 annotations:

```bash
uv run hf download aaa23123/Vispec-stage2 \
  meta.json data.jsonl \
  --repo-type dataset \
  --local-dir "${DATASET_DIR}"
```

Download the LLaVA image archive. `blip_laion_cc_sbu_558k.json` is not needed
because the generated 68,000-row `data.jsonl` already contains the training
conversations:

```bash
uv run hf download liuhaotian/LLaVA-Pretrain \
  images.zip \
  --repo-type dataset \
  --local-dir "${MEDIA_DIR}"

unzip -q "${MEDIA_DIR}/images.zip" -d "${MEDIA_DIR}"
```

Keep `images.zip` until the full image validation below passes. Removing it
after validation requires explicit user approval.

## 6. Download the frozen target

```bash
uv run hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir "${TARGET_DIR}"
```

## 7. Point metadata at the new image directory

The uploaded `meta.json` contains the original server's absolute
`/data/LLaVA-Pretrain` path. Replace only `media_dir` in the downloaded copy:

```bash
jq --arg media_dir "${MEDIA_DIR}" \
  '.vispec_stage2.media_dir = $media_dir' \
  "${DATASET_DIR}/meta.json" \
  > "${DATASET_DIR}/meta.json.tmp"

mv "${DATASET_DIR}/meta.json.tmp" "${DATASET_DIR}/meta.json"
```

Do not modify the tracked recipe YAML just to change machine-local paths. Use
the CLI overrides in the training command below.

## 8. Validate every training input

```bash
set -euo pipefail

test -s "${DATASET_DIR}/meta.json"
test -s "${DATASET_DIR}/data.jsonl"
test -s "${TARGET_DIR}/config.json"

NUM_SAMPLES="$(wc -l < "${DATASET_DIR}/data.jsonl")"
echo "Training samples: ${NUM_SAMPLES}"
test "${NUM_SAMPLES}" -eq 68000

jq -e \
  --arg media_dir "${MEDIA_DIR}" \
  '.vispec_stage2.media_dir == $media_dir' \
  "${DATASET_DIR}/meta.json"

MISSING_IMAGE=""
while IFS= read -r relative_path; do
  if [[ ! -f "${MEDIA_DIR}/${relative_path}" ]]; then
    MISSING_IMAGE="${relative_path}"
    break
  fi
done < <(jq -r '.images[]?' "${DATASET_DIR}/data.jsonl")

if [[ -n "${MISSING_IMAGE}" ]]; then
  echo "Missing image: ${MEDIA_DIR}/${MISSING_IMAGE}" >&2
  exit 1
fi

echo "All 68,000 samples and referenced images are present."
```

Also perform a CPU import/config smoke before occupying the GPU:

```bash
uv run python -c '
from pathlib import Path
import yaml
p = Path("examples/speculative/dflash/qwen2_5_vl_dflash_hard_mrope_full.yaml")
c = yaml.safe_load(p.read_text())
assert c["recipe_args"]["num_epochs"] == 6
assert c["recipe_args"]["micro_batch_size"] == 28
assert c["recipe_args"]["seq_length"] == 3072
assert c["recipe_args"]["block_size"] == 8
assert c["recipe_args"]["num_anchors"] == 256
assert c["recipe_args"]["spatial_rope_mode"] == "replace"
assert c["recipe_args"]["spatial_rope_sections"] == [16, 24, 24]
print("Hard-MRoPE config validated")
'
```

## 9. Start the exact full training run

Before running, tell the user that the reference took about 9 hours 46 minutes
on one B200. Then run the command in the foreground:

```bash
uv run torchrun \
  --standalone \
  --nproc_per_node=1 \
  -m nemo_automodel.recipes.llm.train_dflash \
  --config examples/speculative/dflash/qwen2_5_vl_dflash_hard_mrope_full.yaml \
  --recipe_args.target_model_name_or_path "${TARGET_DIR}" \
  --recipe_args.output_dir "${RUN_DIR}" \
  --dataset.path_or_dataset "${DATASET_DIR}/meta.json" \
  --checkpoint.checkpoint_dir "${RUN_DIR}/checkpoints" \
  --wandb.dir "${RUN_DIR}/wandb" \
  2>&1 | tee "${RUN_DIR}/train.log"
```

The exact run uses a micro-batch size of 28. If this OOMs on different
hardware, do not silently lower it: report the GPU and error to the user.
Changing the micro-batch size changes the number of optimizer steps and is not
an exact reproduction unless the overall batch/step schedule is redesigned.

## 10. Verify completion and checkpoint integrity

```bash
FINAL_CHECKPOINT="${RUN_DIR}/checkpoints/epoch_6_step_14574/model/consolidated"

test -s "${RUN_DIR}/train.log"
test -s "${FINAL_CHECKPOINT}/model.safetensors.index.json"

echo "Final checkpoint: ${FINAL_CHECKPOINT}"
tail -100 "${RUN_DIR}/train.log"
```

Perform a CPU streaming finite-value check without loading the entire draft at
once:

```bash
FINAL_CHECKPOINT="${RUN_DIR}/checkpoints/epoch_6_step_14574/model/consolidated" \
uv run python -c '
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(os.environ["FINAL_CHECKPOINT"])
index = json.loads((root / "model.safetensors.index.json").read_text())
files = sorted(set(index["weight_map"].values()))
tensors = 0
elements = 0
for name in files:
    with safe_open(root / name, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite tensor: {key}")
            tensors += 1
            elements += tensor.numel()
print({"files": len(files), "tensors": tensors, "elements": elements, "finite": True})
'
```

Report the final epoch, global step, elapsed time, checkpoint path, tensor
count, parameter-element count, and any warnings. A complete checkpoint proves
that the run finished technically; it does not prove benchmark improvement.

## 11. GitHub handoff rules

Downloaded data and training outputs are not GitHub artifacts. Before any
commit, verify the remote and inspect every changed path:

```bash
git remote get-url --push origin
git status --short --branch
```

The push URL must be exactly:

```text
https://github.com/jacky566/Automodel_lao.git
```

If source or documentation was intentionally changed, run the repository's
required formatting/tests, stage only explicit code/document paths, and create
a DCO-signed Conventional Commit. Example for this runbook only:

```bash
git add AGENT.md
git diff --cached --check
git diff --cached --name-status
git commit -s -m "docs(speculative): add Hard-MRoPE reproduction runbook"
git push origin main
```

Do not commit if no tracked source or documentation changed. Never stage any
path containing `repro_assets`, `repro_runs`, `checkpoint`, `safetensors`,
`distcp`, `training_data`, model files, image archives, logs, or credentials.

## Final report to the user

At handoff, state:

- source repository and exact commit;
- GPU model and environment setup result;
- dataset repo, sample count, and image validation result;
- target model path;
- exact training command and whether it is still running or complete;
- final checkpoint and training-log paths;
- tests and integrity checks performed;
- Git commit and the verified `origin/main` destination, if repository files
  were changed;
- any deviation from the reference run.
