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

## Relevant current files

- `examples/speculative/dflash/qwen2_5_vl_dflash.yaml`
- `examples/speculative/dflash/qwen2_5_vl_domino.yaml`
- `nemo_automodel/components/speculative/dflash/draft_qwen3.py`
- `nemo_automodel/components/speculative/dflash/domino_core.py`
- `nemo_automodel/components/speculative/dflash/target.py`
- `tools/transformers_vlm_spec_bench.py`
