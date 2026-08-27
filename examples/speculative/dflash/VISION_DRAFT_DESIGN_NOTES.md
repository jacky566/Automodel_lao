# DFlash Visual Conditioning Design Notes

## Current architecture

The current Qwen2.5-VL DFlash draft is a target-conditioned text decoder, not an
independent VLM. The frozen target performs image encoding and multimodal fusion,
then DFlash consumes language-decoder hidden states from target layers 1, 13,
and 25. Those states already include information carried by the image tokens.

Current inference settings:

- Three draft layers with `block_size: 8`.
- BF16 greedy decoding and block target verification.
- No system prompt and a fixed 512-token output in the benchmark.
- No Domino correction head (`projector_type` is unset in the current checkpoint).
- No direct draft-side `pixel_values` input and no second vision encoder.

The current benchmark's mean accepted draft length was approximately 1.185
tokens when this design discussion was recorded. Treat that value as a baseline
observation, not a guaranteed result; rerun the four-benchmark suite before
comparing future variants.

## Recommendation

Do not add a full independent ViT as the first visual extension. It would repeat
image encoding already required by the target, increase parameters, memory, and
prefill latency, and may not improve acceptance enough to increase end-to-end
throughput.

First reuse the target's image-token hidden states through a lightweight visual
conditioning branch:

```text
target image-token hidden states
    -> 2-8 learned queries or attention pooling
    -> Linear + RMSNorm
    -> gated visual K/V context for DFlash
```

The target vision tower and target language model remain frozen. Compute the
compressed visual context once during prefill, cache it, and reuse it for every
draft round. Initialize the fusion gate near zero so an existing DFlash
checkpoint can be warm-started with behavior close to the current model.

This design makes the visual signal explicit without processing `pixel_values`
twice. It is also closer to ViSpec's lightweight image adaptor and two query
tokens than to adding another full vision tower.

## Recommended priority

1. Establish a fresh four-benchmark DFlash baseline and report acceptance and
   phase timing per dataset.
2. Add or train the existing Domino causal correction head. The current
   non-causal block draft lacks full token-to-token dependency inside each
   eight-token block, which may be a larger limitation than visual conditioning.
3. Add a two-query visual resampler over cached target image-token hidden states.
4. Increase visual query count only if two queries underfit; compare 2, 4, and 8.
5. Consider a small independent ViT only if explicit reuse of target visual
   features improves visual-heavy datasets but remains clearly insufficient.
6. Do not duplicate the full Qwen2.5-VL vision tower unless measured throughput
   demonstrates a net benefit.

## Proposed ablation matrix

| Variant | Causal correction | Visual branch | Purpose |
|---|---|---|---|
| A | None | None | Current DFlash baseline |
| B | Domino | None | Measure the block-dependency bottleneck |
| C | None | 2-query cached visual adaptor | Isolate explicit visual conditioning |
| D | Domino | 2-query cached visual adaptor | Recommended combined candidate |
| E | Domino | 4- or 8-query adaptor | Test visual capacity after D |
| F | Domino | Independent Tiny ViT | Last-resort comparison only |

Use the same 100 samples per benchmark, fixed 512-token output, image resolution
bounds, target checkpoint, SDPA backend, and greedy decoding for every variant.

## Metrics and decision rule

Record at least the following per benchmark:

- `accept_length` and `emitted_tokens_per_step`.
- `acceptance_rate`.
- Overall output tokens per second and speedup over cached autoregressive decoding.
- Target prefill, draft, and target verification time.
- Peak GPU memory.
- Exact-match and token-match diagnostics against the same greedy target baseline.

Prioritize final tokens per second, not acceptance in isolation. Keep a visual
extension only when the reduction in target verification time is greater than
its additional prefill and draft cost. Inspect COCO Caption, MM-Vet, TextVQA,
VizWiz, and GQA separately; a gain limited to one dataset should not be reported
as a general DFlash improvement.

Useful diagnosis before implementation:

- Compare acceptance across visually demanding and language-dominant samples.
- Compare the current model with image-token context retained versus masked in
  a controlled diagnostic.
- If acceptance remains similarly low when visual information is unimportant,
  prioritize the causal correction path rather than a vision encoder.

## Implementation boundary

Qwen2.5-VL is a VLM target with a vision tower, projector, processor contract,
and text backbone. Keep generic cached-visual conditioning contracts in the
DFlash component. Any Qwen2.5-VL-specific image-token extraction, module paths,
or processor policy belongs in the Qwen2.5-VL model package rather than in a
generic speculative-decoding layer.

If an independent vision tower creates a new standalone checkpoint architecture,
classify it as a VLM and add the corresponding model package, state-dict adapter,
registry entry, and custom config registration when required. Do not hide target
model identities or vision-module paths in generic DFlash code.

## Required validation before full training

- Tiny CPU shape tests for image-token selection, resampling, projection, and
  gated fusion.
- A gate-zero equivalence test showing the new model reproduces the existing
  DFlash path before visual conditioning is enabled.
- Forward and backward tests with image-text inputs and finite gradients.
- A test proving visual context is computed once during prefill and reused
  without stale cache state across prompts.
- Save/reload coverage, including deterministic initialization of new weights
  and an explicit migration path from the current DFlash checkpoint.
- State-dict mapping and component parity tests if a new vision tower or
  rewritten vision layer is introduced.
- A short training test followed by the complete A/B benchmark matrix above.

## Two-stage training workflow

The implemented two-query workflow warm-starts the original 14,574-step DFlash
checkpoint and gives each training stage one complete pass over the 68,000-row
dataset. With a micro-batch size of 28 and one GPU, this is normally 2,429
optimizer steps per stage (4,858 total):

| Stage | Trainable parameters | Epochs | Expected steps | Learning rates |
|---|---|---:|---:|---|
| 1 | Per-layer visual resampler, cross-attention, and gate | 1 | 2,429 | visual `4e-4` |
| 2 | DFlash backbone plus all visual modules | 1 | 2,429 | backbone `1e-4`, visual `3e-4` |

Each of the three draft layers compresses image-token features from its paired
target layer into two 256-dimensional queries using four attention heads. This
keeps the visual branch to roughly 9.2 million parameters instead of performing
full-width 3,584-dimensional cross-attention. The resulting contexts are built
once in inference prefill and passed explicitly through subsequent draft
rounds; they are not stored as mutable model state, so prompts cannot reuse
stale context.

The cached context pools all image tokens in one sample. Therefore every image
must occur before the first supervised assistant token. Training rejects later
images instead of allowing future-turn visual leakage. The current 68,000-row
ViSpec training set satisfies this rule (all rows were audited), but packed or
multi-turn datasets must be checked before use.

The local Transformers evaluator implements the visual adaptor. The current
vLLM DFlash runtime does not; `serve_vllm.py` rejects these checkpoints so it
cannot silently serve only the text backbone.

Run both stages in sequence:

```bash
DFLASH_NPROC=1 bash scripts/train_qwen2_5_vl_dflash_visual.sh
```

The stage configurations are
`qwen2_5_vl_dflash_visual_stage1.yaml` and
`qwen2_5_vl_dflash_visual_stage2.yaml`. The script resolves relative checkpoint
and output paths from the repository root. Adjust the `/data/models` and dataset
paths before starting if the server mount differs.

Stage 1 saves a resumable checkpoint every 400 optimizer steps, and Stage 2
saves one every 1,000 steps. Both stages always save a final consolidated
safetensors checkpoint; retention keeps the two most recent checkpoints in each
stage directory. The sequential launcher validates the final Stage 1
safetensors and passes its resolved path directly into Stage 2.

Before the full run, both stages were exercised for one optimizer step with a
`micro_batch_size` of 28 and `seq_length` of 3,072 on one NVIDIA B200. The final
configs use 256 anchors in both stages, matching the original DFlash training
objective. Stage transitions, consolidated checkpoint reload, `torch.compile`,
and one cached-visual ScienceQA inference sample were also verified.

## Relevant current files

- `examples/speculative/dflash/qwen2_5_vl_dflash.yaml`
- `examples/speculative/dflash/qwen2_5_vl_domino.yaml`
- `nemo_automodel/components/speculative/dflash/draft_qwen3.py`
- `nemo_automodel/components/speculative/dflash/domino_core.py`
- `nemo_automodel/components/speculative/dflash/target.py`
- `tools/transformers_vlm_spec_bench.py`

## Target-layer routing smoke experiment (2026-08-19)

This minimum-cost diagnostic tests whether the original shared projection of
target layers 1, 13, and 25 hides useful layer-specific multimodal information.
It warm-starts the matched 400-step DFlash checkpoint and adds one scalar gate
per draft layer. Draft layer `i` receives

```text
shared_context + tanh(gate_i) * (paired_target_layer_i - shared_context)
```

after applying the corresponding slice of the existing `fc.weight`. All
backbone weights remain frozen, the gates start at zero, and the zero-gate path
is bitwise identical to the original DFlash output. This changes neither the
context sequence length nor the draft KV-cache size.

Training used the 1,280-row matched VLM dataset, 256 anchors, block size 8,
micro-batch size 28, BF16, and 100 optimizer steps. Only three scalars were
trained with AdamW, peak LR `5e-3`, 5% warmup, and cosine decay. The run took
3 minutes 31 seconds on one NVIDIA B200. Final gate strengths were
`[-0.0124, -0.0225, +0.0302]` for target layers `[1, 13, 25]`.

The evaluation reused the same first 10 samples, fixed 256 output tokens,
greedy block verification, block size 8, and SDPA target backend as the matched
DFlash baseline:

| Benchmark | Variant | Accepted draft tokens / verify step | Emitted tokens / verify step | Acceptance rate | Tokens/s |
|---|---|---:|---:|---:|---:|
| GQA | Matched DFlash | 0.9076 | 1.9076 | 13.16% | 60.46 |
| GQA | + layer routing | 0.9033 | 1.9033 | 13.10% | 68.00 |
| TextVQA | Matched DFlash | 1.0205 | 2.0205 | 14.80% | 64.45 |
| TextVQA | + layer routing | 1.0221 | 2.0221 | 14.82% | 67.80 |

The acceptance changes are -0.47% on GQA and +0.16% on TextVQA, so this does
not provide evidence of a useful improvement. The throughput values came from
one timing run per checkpoint and should not be interpreted as a routing gain:
the routing adds computation, acceptance is effectively unchanged, and the
target-verification timing differed between runs. The result instead suggests
that simple per-draft-layer selection of already-captured target features is
not the main VLM bottleneck. The next targeted diagnostic should preserve and
route Qwen2.5-VL's multimodal position structure (MRoPE), rather than adding
more visual pooling capacity or extending this scalar-routing run.

Artifacts:

- Config: `qwen2_5_vl_dflash_layer_routing_smoke.yaml`
- Checkpoint: `dflash_layer_routing_smoke_checkpoints/epoch_2_step_100/model/consolidated`
- Results: `benchmark_results/dflash_layer_routing_step100_gqa_10x256.json`
  and `benchmark_results/dflash_layer_routing_step100_textvqa_10x256.json`

## MRoPE gate and direct-replacement experiments (2026-08-26)

The next diagnostic preserves the target VLM's temporal, height, and width
positions in draft attention. Each draft layer owns one zero-initialized scalar
that interpolates its rotated query/key tensors between the existing one-axis
RoPE path and the target-compatible MRoPE path. Gate zero is bitwise equivalent
to matched DFlash, and the change does not add tokens or increase KV-cache
length.

The target wrapper computes and returns the same three-axis positions used by
the frozen Qwen2.5-VL target. Training constructs each sampled draft block's
positions from its anchor's three-axis position. Inference extends the final
prompt position equally along all three axes for generated text tokens. The
MRoPE channel sections are `[16, 24, 24]`, matching the target's 128-dimensional
attention head.

CPU coverage includes zero-gate exact equivalence, text-position equivalence,
numerical parity with Transformers' Qwen2.5-VL MRoPE reference, target position
capture, block-position construction, gate-only gradients, and checkpoint
migration. The focused suite passes 114 tests. A real one-step B200 smoke run
also completed training, saved consolidated safetensors, updated all three gates
to finite nonzero values, and generated one fixed-length 256-token GQA sample
from the exported checkpoint.

The complete diagnostic trains only three gates for 100 optimizer steps on the
1,280-row matched dataset, saving consolidated checkpoints at steps 50 and 100.
The launcher then evaluates all five legacy benchmarks with ten samples and 256
fixed output tokens:

```bash
bash scripts/run_qwen2_5_vl_dflash_spatial_rope_100step.sh
```

The gated run completed without an acceptance gain. Its final strengths were
`[-0.00365, -0.00190, -0.00294]`, and its five-task mean accepted draft length
was `0.9012`, compared with `0.9064` for matched DFlash. The near-zero gates
show that a frozen one-axis backbone does not benefit from a small local MRoPE
interpolation, but this result alone does not test whether the attention
backbone can co-adapt to direct MRoPE.

A controlled direct-replacement experiment therefore removed the gate,
replaced one-axis RoPE with MRoPE in every draft layer, and trained the complete
draft backbone. Both direct-MRoPE and matched DFlash started from the same
original `epoch_6_step_14574` checkpoint and used the same 1,280 examples, 400
steps, peak learning rate `3e-5`, 256 anchors, block size 8, and cosine schedule.
Checkpoints at steps 200 and 400 were each evaluated on the same first ten
samples from all five legacy benchmarks with 256 fixed output tokens.

| Benchmark | Matched DFlash | Hard MRoPE step 200 | Hard MRoPE step 400 | Step-400 change |
|---|---:|---:|---:|---:|
| GQA | 0.9121 | 0.9753 | 0.9799 | +0.0678 |
| TextVQA | 1.0292 | 1.0253 | 1.0221 | -0.0071 |
| COCO Caption | 1.0086 | 1.0205 | 1.0205 | +0.0119 |
| CharXiv Reasoning | 0.7273 | 0.7204 | 0.7181 | -0.0092 |
| MMMU-Pro | 0.8548 | 0.8810 | 0.8837 | +0.0290 |
| **Mean** | **0.9064** | **0.9245** | **0.9249** | **+0.0185** |

Direct MRoPE is materially different from the failed scalar gate: it improves
GQA and MMMU-Pro after full-backbone adaptation. However, the mean gain remains
below the predeclared `+0.03` threshold, TextVQA and CharXiv regress slightly,
and another 200 steps add only `+0.00037` mean acceptance. This is evidence of a
task-specific spatial benefit, not a general replacement for matched DFlash.
Do not spend the original 14,574-step training budget on a random-initialized
MRoPE draft yet. The next experiment should test a region-aware,
token-conditioned path on the GQA/OCR split, where the spatial signal can be
isolated more directly.

Artifacts:

- Config: `qwen2_5_vl_dflash_hard_mrope_400step.yaml`
- Checkpoints: `dflash_hard_mrope_400step_checkpoints/epoch_4_step_200/model/consolidated`
  and `dflash_hard_mrope_400step_checkpoints/epoch_8_step_400/model/consolidated`
- Results: `benchmark_results/dflash_hard_mrope_step200_legacy5_10x256.json`
  and `benchmark_results/dflash_hard_mrope_step400_legacy5_10x256.json`
