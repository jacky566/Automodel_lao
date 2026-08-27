# Why VLM speculative acceptance is lower than LLM acceptance

Analysis date: 2026-08-19

## Technical summary

The current evidence points to a combination of causes rather than an inherent inability to process images. The
largest verified and actionable issue is training-data mismatch: all 68,000 rows in the current stage-2 dataset are
single-image prompts that explicitly request at least 1,000 words, with answers averaging 595.93 words. The
evaluation suite instead mixes short visual question answering, OCR, captioning, and reasoning. This narrow long-form
caption distribution is unlikely to teach the draft the token transitions that dominate GQA, TextVQA, MMMU-Pro, and
CharXiv.

VLM decoding is also intrinsically harder for exact-token speculative acceptance. Visual evidence can support several
plausible descriptions, OCR and spatial details create high-entropy decisions, and one early disagreement changes the
entire continuation. Finally, the current DFlash path carries visual information only indirectly through selected
target-language hidden states. It does not omit image tokens, but it compresses three target layers into one context
and applies a Qwen3-style draft with one-dimensional rotary positions to a flattened Qwen2.5-VL context whose target
uses multimodal RoPE. That contract is a plausible framework bottleneck and requires controlled ablation.

The first six controls have now been run. Text-only acceptance did not rise above the VLM mean, while removing image
token hidden states from only the draft caused a large acceptance drop. Reordering those same states had almost no
effect. A 400-step task-matched warm-start continuation then improved mean acceptance only from 1.8833 to 1.9064,
below the predeclared 0.03-0.05 threshold, with most of the gain concentrated on GQA. Training mismatch therefore
matters, but is not sufficient to explain the main gap. Position-wise measurement shows that acceptance is highest in
the first 64 generated tokens and generally falls later; matched continuation leaves tokens 1-32 unchanged and puts
its gain mainly after token 32. The next experiment should therefore target task-dependent late-generation entropy and
block-internal dependency before adding a spatial visual module. The Domino follow-up confirms that its causal head
substantially improves proposal offsets 2-7, but the separately trained Domino backbone is worse at offset 1 and on
late GQA/COCO continuations. The next training experiment should preserve the stronger DFlash backbone while adding
the Domino head, rather than train another draft from scratch.

## Evidence: the current data is narrowly matched to long descriptions

| Dataset property | Observed value | Diagnostic implication |
|---|---:|---|
| Rows | 68,000 | Adequate volume for a draft continuation experiment |
| Images | 68,000; one per row | No text-only control distribution |
| Prompts requesting at least 1,000 words | 68,000 (100%) | Severe prompt-length and task-style concentration |
| Mean answer length | 595.93 words | Training is dominated by long-form continuation |
| Answers longer than 500 words | 49,074 (72.17%) | Most optimization tokens come from very long continuations |
| Training epochs | 6 | Repeats the same narrow distribution rather than adding task coverage |
| Evaluation tasks | GQA, TextVQA, COCO Caption, CharXiv, MMMU-Pro | Includes OCR, short answers, captions, and reasoning |

The most common prompt families are variants of “briefly describe/summarize the image,” even though every prompt then
adds the contradictory requirement to answer with at least 1,000 words. The first inspected rows also contain verbose,
repetitive, and occasionally unsupported image descriptions. This is useful as caption pretraining, but it is not a
representative speculative-training distribution for the current benchmark mix.

## Evidence: visual information is present but only indirectly represented

The target Qwen2.5-VL model performs image encoding and multimodal fusion. DFlash captures target hidden states from
layers 1, 13, and 25 over the full multimodal prompt, including image-token positions. Those states are concatenated,
projected through one linear layer, and used as key/value context by the three-layer Qwen3-style draft. Consequently,
the base draft is not image-blind in the literal sense.

However, the current contract has three limitations:

1. All selected target layers are concatenated and compressed into one shared context before the draft layers consume
   them, which can blur layer-specific visual information.
2. Qwen2.5-VL derives three-axis multimodal positions for image tokens, while the Qwen3 draft applies its own
   one-dimensional rotary positions to the flattened context. The target hidden states already contain spatial
   information, but the draft attention does not preserve the target's native MRoPE geometry explicitly.
3. The base draft receives no direct region-token or pixel contract. Visual evidence must survive the target hidden
   states, the three-layer concatenation, and the shared projection.

The failed Q-Former and pooled-MLP experiments do not prove that explicit visual conditioning is unnecessary. They
show that compressing the entire image into two queries or one global vector and adding it through a gate does not
improve this draft. Both approaches can discard precisely the local OCR and spatial information needed by the harder
benchmarks.

## Current acceptance pattern is task-dependent

Acceptance length here includes the one target-guaranteed token.

| Benchmark | DFlash | Domino | Domino change |
|---|---:|---:|---:|
| GQA | 1.8083 | 1.6106 | -0.1978 |
| TextVQA | 2.0228 | 2.2528 | +0.2300 |
| COCO Caption | 2.0023 | 1.9884 | -0.0140 |
| CharXiv Reasoning | 1.7413 | 1.9155 | +0.1742 |
| MMMU-Pro | 1.8415 | 1.9483 | +0.1069 |
| **Five-task mean** | **1.8833** | **1.9431** | **+0.0599** |

Domino's improvement on TextVQA, CharXiv, and MMMU-Pro shows that missing causal dependency inside an eight-token
parallel block is one real bottleneck. Its regression on GQA shows that causal correction alone does not solve the
whole VLM gap. ViSpec tree acceptance is not directly comparable because it uses a different proposal topology and
denominator.

## Test 1 result: same-target text-only control

The same Qwen2.5-VL-7B target, original DFlash checkpoint, SDPA engine, block verifier, ten samples per task, and fixed
256-token output were used. Acceptance length includes the one target-guaranteed token.

| Text benchmark | Acceptance length | Acceptance rate | Throughput (tok/s) |
|---|---:|---:|---:|
| MT-Bench | 1.6846 | 9.78% | 62.89 |
| HumanEval | 1.5130 | 7.33% | 55.76 |
| GSM8K | 2.0922 | 15.60% | 75.81 |
| Alpaca | 1.7507 | 10.72% | 62.05 |
| **Four-task mean / aggregate throughput** | **1.7601** | — | **63.35** |

The text-only mean is 0.1231 below the legacy VLM mean of 1.8833. This does not show that images make speculative
decoding easier; the task distributions are not matched. It does reject the simple explanation that removing images
automatically restores high LLM-like acceptance. Draft quality and task/training-distribution match remain major
drivers: GSM8K is substantially easier for this checkpoint than HumanEval even though both are text-only.

## Test 2 result: image-token hidden-state ablation

The frozen target always saw the correct image. Only image-token rows in the target hidden states entering the draft
prompt cache were changed. `keep` is the existing original-DFlash run; `zero` removes those rows' information while
preserving sequence length; `shuffle` reverses the image rows within each prompt while preserving their values.

| VLM benchmark | Keep | Zero | Zero change | Shuffle | Shuffle change |
|---|---:|---:|---:|---:|---:|
| GQA | 1.8083 | 1.6478 | -0.1605 | 1.8032 | -0.0051 |
| TextVQA | 2.0228 | 1.5307 | -0.4921 | 2.0149 | -0.0079 |
| COCO Caption | 2.0023 | 1.7082 | -0.2941 | 2.0008 | -0.0016 |
| CharXiv Reasoning | 1.7413 | 1.4394 | -0.3020 | 1.7355 | -0.0059 |
| MMMU-Pro | 1.8415 | 1.5912 | -0.2503 | 1.8401 | -0.0013 |
| **Five-task mean** | **1.8833** | **1.5835** | **-0.2998** | **1.8789** | **-0.0044** |

Clearing image rows lowers mean acceptance by 15.9%, so base DFlash already makes material use of explicit visual
hidden states. The claim that visual tokens are absent or wholly ignored is rejected. In contrast, reversing image-row
order changes the mean by only 0.23%. This indicates that the current draft benefits mainly from the presence and
aggregate content of visual features but is nearly insensitive to their flattened spatial order. That is consistent
with a framework limitation in preserving visual geometry, and explains why compressing the same information into a
global Q-Former or pooled-MLP vector did not add useful signal.

Aggregate throughput was 61.15 tok/s for the prior keep run, 52.01 tok/s for zero, and 64.81 tok/s for shuffle. Only
one timing run was made per ablation, so the keep-versus-shuffle timing difference should be treated as run variance;
the acceptance comparison is the primary result.

## Test 3 result: 400-step matched-data warm-start continuation

The original six-epoch DFlash checkpoint was warm-started without adding any visual adaptor. A deterministic training
set was built from 256 non-benchmark images, with five target-generated prompt types per image: short VQA, OCR,
captioning, visual reasoning, and long description (1,280 rows total). Training used sequence length 2,048,
micro-batch size 28, block size 8, 256 anchors, learning rate 3e-5, and checkpoints at steps 200 and 400. The run took
about 13 minutes and skipped no samples.

Evaluation reused the original legacy5 setup exactly: the same target, SDPA inference, block verification, ten samples
per benchmark, and fixed 256-token output. Acceptance length includes the one target-guaranteed token.

| Benchmark | Original | Step 200 | Change | Step 400 | Change |
|---|---:|---:|---:|---:|---:|
| GQA | 1.8083 | 1.8979 | +0.0896 | 1.9121 | +0.1038 |
| TextVQA | 2.0228 | 2.0276 | +0.0048 | 2.0292 | +0.0064 |
| COCO Caption | 2.0023 | 2.0055 | +0.0031 | 2.0086 | +0.0063 |
| CharXiv Reasoning | 1.7413 | 1.7331 | -0.0082 | 1.7273 | -0.0141 |
| MMMU-Pro | 1.8415 | 1.8534 | +0.0120 | 1.8548 | +0.0133 |
| **Five-task mean** | **1.8833** | **1.9035** | **+0.0203** | **1.9064** | **+0.0231** |
| **Aggregate throughput (tok/s)** | **61.15** | **64.67** | **+3.52** | **64.57** | **+3.42** |

Step 400 is the better checkpoint by mean acceptance, but its +0.0231 gain is below the predeclared +0.03-0.05
criterion for treating training distribution as the main actionable driver. The task split helps GQA substantially,
but barely changes TextVQA and COCO, slightly improves MMMU-Pro, and regresses CharXiv. This pattern is evidence that
the old long-form-only distribution was one contributor, not the dominant general explanation. The throughput values
come from one run per checkpoint and are not sufficient to claim a stable speedup; the acceptance result is primary.

## Test 4 result: block size 4 versus 8

The original and step-400 checkpoints were evaluated again with inference block size 4. All other legacy5 parameters
were unchanged. `Acceptance length` remains emitted tokens per verification step and includes the guaranteed token;
`acceptance rate` uses the number of available proposal slots as its denominator, so it should not be compared directly
between different block sizes.

| Checkpoint | Block size | Mean acceptance length | Mean acceptance rate | Aggregate throughput (tok/s) |
|---|---:|---:|---:|---:|
| Original | 8 | 1.8833 | 12.62% | 61.15 |
| Original | 4 | 1.8313 | 27.71% | 64.71 |
| Matched step 400 | 8 | **1.9064** | 12.95% | **64.57** |
| Matched step 400 | 4 | 1.8507 | 28.36% | 62.14 |

Reducing the proposal block raises the normalized acceptance rate, as expected from predicting fewer future slots,
but reduces the actual number of tokens emitted per verifier call for both checkpoints. It improves the original
checkpoint's single-run throughput by 5.8%, yet reduces the step-400 checkpoint's throughput by 3.8%. The inconsistent
speed direction means block size 4 is not a general fix. With the current evidence, matched step 400 plus block size 8
is the best overall setting. Timing should be repeated before making a production speed claim.

## Test 5 result: position-wise acceptance

The evaluator now records every verification round's one-based generated-token anchor and aggregates rounds into
tokens 1-32, 33-64, 65-128, and 129-256. Each bucket reports the same metric used elsewhere: accepted draft tokens per
verification round plus the one guaranteed target token. A round crossing a boundary is assigned to the bucket that
contains its anchor.

This instrumentation also fixed an old measurement boundary issue: candidates in the final partial block beyond the
requested 256 output tokens are no longer counted. The corrected absolute five-task means are 1.8782 for the original
checkpoint and 1.9013 for matched step 400. The continuation gain remains +0.0231, so the previous conclusion is
unchanged; older tables in this document predate this exact-boundary correction.

| Generated position | Original | Matched step 400 | Change |
|---|---:|---:|---:|
| 1-32 | 2.0624 | 2.0533 | -0.0091 |
| 33-64 | 2.0750 | 2.0968 | +0.0218 |
| 65-128 | 1.8831 | 1.9026 | +0.0195 |
| 129-256 | 1.7807 | 1.8114 | +0.0308 |

The first 64 tokens are already the easiest region for the original checkpoint. Acceptance then drops by 0.2943 from
tokens 33-64 to 129-256. Matched continuation does not improve tokens 1-32; its largest aggregate gain is in tokens
129-256. This rejects the simple prediction that weak initial visual grounding is the dominant source of the current
gap. It instead supports task-dependent continuation entropy or insufficient dependency modeling later in generation.

The effect is not uniform across tasks. Values below are matched step 400 minus original acceptance length.

| Benchmark | 1-32 | 33-64 | 65-128 | 129-256 |
|---|---:|---:|---:|---:|
| GQA | -0.0472 | +0.0607 | +0.0691 | +0.1475 |
| TextVQA | +0.0017 | -0.0012 | +0.0151 | +0.0045 |
| COCO Caption | -0.0127 | +0.0123 | +0.0100 | +0.0060 |
| CharXiv Reasoning | 0.0000 | +0.0058 | -0.0141 | -0.0208 |
| MMMU-Pro | +0.0012 | +0.0323 | +0.0081 | +0.0144 |

Nearly all of the large late-position gain comes from GQA. TextVQA and COCO barely change, while CharXiv regresses in
the second half. Therefore the matched checkpoint has not learned a general visual improvement; it has learned a
task-specific continuation improvement. With only ten samples per task, small differences should not be interpreted
as statistically stable, but the aggregate early-versus-late direction is large enough to guide the next diagnostic.

## Test 6 result: Domino position and proposal-offset acceptance

The existing Domino checkpoint was evaluated with the same target, SDPA verifier, legacy5 prompts, ten samples per
task, fixed 256-token output, block size 8, corrected final-block accounting, and position instrumentation. The
original and matched DFlash checkpoints were rerun with the same code so proposal offsets are directly comparable.

| Checkpoint | Mean acceptance length | Aggregate throughput (tok/s) |
|---|---:|---:|
| Original DFlash | 1.8782 | 63.95 |
| Matched DFlash step 400 | 1.9013 | 61.01 |
| Domino | **1.9382** | 57.59 |

Domino raises mean acceptance by 0.0600 over original DFlash and 0.0369 over matched DFlash, but the current reference
runtime is slower. PyTorch reports that the GRU weights are non-contiguous and repacks them on calls, so this single
throughput run should not be treated as the optimized cost of the mechanism.

| Benchmark | Original DFlash | Matched DFlash | Domino | Domino vs original |
|---|---:|---:|---:|---:|
| GQA | 1.8041 | 1.9076 | 1.6080 | -0.1960 |
| TextVQA | 2.0142 | 2.0205 | 2.2397 | +0.2256 |
| COCO Caption | 1.9984 | 2.0047 | 1.9845 | -0.0139 |
| CharXiv Reasoning | 1.7379 | 1.7239 | 1.9133 | +0.1754 |
| MMMU-Pro | 1.8364 | 1.8497 | 1.9453 | +0.1088 |

Domino is not a uniform VLM improvement. It substantially helps TextVQA, CharXiv, and MMMU-Pro, is neutral on COCO,
and strongly regresses GQA. Its position curve exposes where the aggregate gain and regression occur.

| Generated position | Original DFlash | Matched DFlash | Domino |
|---|---:|---:|---:|
| 1-32 | 2.0624 | 2.0533 | **2.3103** |
| 33-64 | 2.0750 | 2.0968 | **2.2551** |
| 65-128 | 1.8831 | 1.9026 | **2.0836** |
| 129-256 | 1.7807 | **1.8114** | 1.7076 |

Domino clearly recovers dependency modeling through token 128, but its late GQA and COCO regressions pull the
129-256 aggregate below both DFlash checkpoints. The problem is therefore task-distribution-sensitive rather than a
universal failure at long positions.

Proposal offset `k` is accepted only when the target accepts the entire prefix through the `k`-th draft token. The
rates below aggregate all five tasks.

| Proposal offset | Original DFlash | Matched DFlash | Domino |
|---:|---:|---:|---:|
| 1 | **55.97%** | **56.48%** | 48.84% |
| 2 | 19.82% | 21.19% | **23.53%** |
| 3 | 7.93% | 8.38% | **11.01%** |
| 4 | 2.90% | 2.81% | **5.17%** |
| 5 | 0.82% | 0.91% | **2.45%** |
| 6 | 0.19% | 0.23% | **1.05%** |
| 7 | 0.05% | 0.03% | **0.36%** |

This directly verifies the intended Domino mechanism: causal correction improves offsets 2-7, with the relative gain
growing at farther offsets. Offset 1 uses the base head under the checkpoint's `pure_draft_prefix_len=1`, so its 7.1
percentage-point regression is evidence that the independently trained Domino backbone is weaker, not that the causal
head damaged that offset at inference. A stronger experiment is therefore to warm-start the matched DFlash backbone,
train the Domino head first, then perform a short low-learning-rate joint stage while retaining base-head supervision.

## Test 7 result: matched-DFlash Domino warm-start

The matched step-400 DFlash checkpoint was used to initialize a new Domino model. Direct warm-start requires
`shift_label=false`: with this mapping proposal offset 1 continues to use DFlash block position 1, while the causal
correction begins at offset 2 because `pure_draft_prefix_len=1`. A one-step A/B smoke test produced exactly 92 accepted
offset-1 proposals for both the source DFlash model and warm-started Domino model on the same GQA sample.

Training used the same 1,280-example matched multimodal dataset as Test 3. Both stages used sequence length 2,048,
micro-batch size 28, 256 anchors, block size 8, three draft layers, a 256-dimensional correction projection, and a
1,024-dimensional GRU. Stage 1 trained only the four Domino-head tensors for 200 steps at learning rate `1e-4`; it
finished in 5m58s. Stage 2 jointly trained the draft and head for 200 steps at learning rate `1e-5`, with base-loss
weight starting at 0.5 and decaying through training; it finished in 6m59s. Each stage saved at step 200.

The evaluator's original one-token cuDNN GRU calls repeatedly repacked non-contiguous GRU weights. Replacing those
calls with the mathematically equivalent bias-free single-step recurrence preserved all acceptance counts and raised
the stage-1 GQA throughput from 49.91 to 61.29 tok/s in a direct before/after run. The optimized recurrence is used for
the results below.

| Checkpoint | Mean acceptance length | Aggregate throughput (tok/s) | Timing note |
|---|---:|---:|---|
| Matched DFlash step 400 | 1.9013 | 61.01 | One reference run |
| Old separately trained Domino | 1.9382 | 57.59 | Unoptimized GRU |
| Warm-start Domino stage 1 | **1.9684** | **63.80 +/- 0.35** | Mean and sample SD of three runs |
| Warm-start Domino stage 2 | 1.9653 | 57.49 | One run; target verification was also slower |

Stage 1 improves mean acceptance by 0.0671 (+3.5%) over matched DFlash. Its three aggregate throughput measurements
were 64.10, 63.91, and 63.41 tok/s, giving a mean gain of 4.6% over the matched-DFlash reference run. Stage 2 does not
improve acceptance and was therefore not repeated; because stage 1 and stage 2 have the same dense compute graph and
the stage-2 target verifier also slowed, its single throughput value is not evidence that the weights themselves make
the architecture slower.

| Benchmark | Matched DFlash | Domino stage 1 | Change | Stage-1 throughput, 3-run mean (tok/s) |
|---|---:|---:|---:|---:|
| GQA | 1.9076 | 1.8273 | -0.0803 | 58.85 |
| TextVQA | 2.0205 | 2.0000 | -0.0205 | 64.62 |
| COCO Caption | 2.0047 | 2.0984 | +0.0937 | 69.95 |
| CharXiv Reasoning | 1.7239 | 2.0016 | +0.2777 | 65.18 |
| MMMU-Pro | 1.8497 | 1.9147 | +0.0650 | 61.58 |

| Proposal offset | Matched DFlash | Domino stage 1 | Stage 2 |
|---:|---:|---:|---:|
| 1 | 56.48% | **59.01%** | 59.01% |
| 2 | 21.19% | **26.22%** | 26.01% |
| 3 | **8.38%** | 7.59% | 7.50% |
| 4 | 2.81% | **2.94%** | 2.92% |
| 5 | **0.91%** | 0.89% | 0.89% |
| 6 | **0.23%** | 0.22% | 0.22% |
| 7 | 0.03% | **0.06%** | 0.06% |

Warm-starting fixes the old Domino checkpoint's weak offset-1 backbone and yields a useful offset-2 gain. The remaining
improvement is not uniform: CharXiv, COCO, and MMMU-Pro improve, while GQA and TextVQA regress. Joint fine-tuning adds
no value at this scale, so stage 1 is the selected checkpoint. The next training change should target the GQA/OCR task
split or its spatial conditioning rather than extending generic joint training.

## Driver assessment

| Candidate driver | Assessment | Confidence | Reason |
|---|---|---:|---|
| Training-data/task mismatch | Real but secondary contributor | High | Matched continuation improved mean acceptance by only 0.0231, below the predeclared 0.03-0.05 threshold |
| VLM exact-token prediction is intrinsically harder | Possible background contributor | Medium | Text-only control also has low and highly task-dependent acceptance, so image difficulty alone does not explain the gap |
| Current draft loses or distorts visual structure | Supported contributor | High | Zeroing image rows hurts by 15.9%, while reversing their order changes acceptance by only 0.23% |
| Missing dependency inside the parallel block | Supported contributor | High | Domino improves proposal offsets 2-7 and raises five-task mean acceptance by 0.0600 over original DFlash |
| Weak old Domino base backbone | Confirmed and mitigated | High | Warm-start raises offset 1 from 48.84% to 59.01% and improves mean acceptance to 1.9684 |
| Visual tokens are entirely absent from DFlash | Rejected | High | Full multimodal target hidden states are passed into draft attention |
| A larger global visual adaptor will fix the gap | Not supported | High | Two-query Q-Former and pooled-MLP gate experiments did not improve acceptance |

## Smallest experiments that separate the causes

### 1. Same-target text-only control — completed

Run the Qwen2.5-VL target and the same DFlash/Domino checkpoints on text-only prompts, using the same engine, block size,
and 256-token output. This controls model family and runtime while removing image-token complexity.

- If text-only acceptance rises to the expected LLM range, the remaining gap is visual/task-specific.
- If it stays low, first investigate training distribution, target-layer selection, position handling, and checkpoint
  quality rather than adding a visual module.

### 2. Image-token context ablation — completed

Keep the correct image in the frozen target, but mask or shuffle only the image-token rows of `target_hidden` before
they enter the draft. The target verifier must still see the original image.

- A large acceptance drop proves that base DFlash already uses image-token context.
- Little or no change means the draft largely ignores explicit image rows and relies on language-token hidden states or
  language priors.

This is more diagnostic than adding another adaptor because it directly measures whether the existing visual path is
causal for acceptance.

### 3. Position-wise and task-wise acceptance — completed

Acceptance is highest in the first 64 tokens and declines later. Matched continuation improves later GQA positions but
does not improve the first 32 tokens or generalize to CharXiv. This does not support initial visual grounding as the
single dominant bottleneck.

### 4. Small matched-data continuation — completed

Warm-start the original DFlash checkpoint and train for only 300-500 optimizer steps on a held-out, non-benchmark mix
of short VQA, OCR, caption, and visual-reasoning responses generated by the same target model. Do not start from
scratch and do not add an adaptor in this test.

Use a starting task mix such as:

- 40% short VQA/OCR responses, mostly below 64 tokens.
- 20% concise captions, roughly 64-128 tokens.
- 20% visual reasoning, roughly 128-256 tokens.
- 20% longer descriptions, retained to avoid catastrophic narrowing.

The measured improvement was +0.0231 at step 400. This is positive but below the decision threshold, so data mismatch
is not the main actionable driver. Framework-level visual/positional changes and decoding-policy tests take priority.

### 5. Block-size 4 diagnostic — completed

The normalized acceptance rate increased, but emitted tokens per verification step fell. Throughput improved only for
the original checkpoint and regressed for matched step 400, so four-token proposals are not a robust default.

## Recommended next steps

1. Use the head-only warm-start checkpoint as the current Domino candidate. Do not use the joint stage: it slightly
   reduces mean acceptance and does not improve any benchmark materially.
2. Add or rebalance a small GQA/OCR-spatial training slice and evaluate it against the current stage-1 checkpoint. Keep
   the matched DFlash offset-1 rate and the stage-1 offset-2 gain as explicit guardrails.
3. If GQA/OCR spatial subsets remain weak after this targeted data test, implement a region-aware,
   token-conditioned visual path that preserves multiple image regions and explicit spatial positions; do not use
   another global compression vector.
4. Do not extend matched-data training beyond 400 steps yet: the gain from step 200 to step 400 was only +0.0029.
5. Keep block size 8 for the current best checkpoint; repeat throughput timing only if a production configuration is
   needed.
6. Do not add a full ViT or repeat the global Q-Former/pooled-MLP experiments at this stage.

## Scope and limitations

The dataset profile uses the current 68,000-row stage-2 JSONL and repository configuration. Architecture findings are
based on the current local DFlash implementation and Qwen2.5-VL checkpoints. The text and VLM benchmark task
distributions are not semantically matched, so their mean difference is not a causal estimate of image difficulty.
Each ablation uses ten samples per task; a larger run is required before interpreting small per-task differences.
