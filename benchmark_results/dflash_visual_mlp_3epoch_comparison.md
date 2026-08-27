# Qwen2.5-VL speculative decoding comparison: pooled MLP

Evaluation date: 2026-08-18

## Evaluation setup

- Target: `/data/models/Qwen2.5-VL-7B-Instruct`
- New draft: `dflash_visual_mlp_3epoch_checkpoints/epoch_3_step_7287/model/consolidated`
- Engine: Transformers, batch size 1, SDPA
- DFlash verification: block verification, block size 8
- Benchmarks: GQA, TextVQA, COCO Caption, CharXiv Reasoning, and MMMU-Pro
- Workload: 10 samples per benchmark, exactly 256 output tokens per sample
- Hardware: one NVIDIA B200
- Speedup reference: the same cached autoregressive baseline in
  `benchmark_results/baseline_legacy5_10x256.json`

The matched baseline, original-DFlash, and pooled-MLP runs each completed all 50 samples and generated 12,800 output
tokens. Their raw results are in:

- `benchmark_results/baseline_legacy5_10x256.json`
- `benchmark_results/dflash_legacy5_10x256.json`
- `benchmark_results/dflash_visual_mlp_3epoch_legacy5_10x256.json`

## Pooled MLP versus original DFlash

| Benchmark | Original accept length | MLP accept length | Accept delta | Original tok/s | MLP tok/s | Throughput delta | Original speedup | MLP speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GQA | 0.8083 | 0.8070 | -0.0013 | 58.32 | 57.62 | -1.19% | 1.602x | 1.583x |
| TextVQA | 1.0228 | 1.0181 | -0.0048 | 66.35 | 67.08 | +1.11% | 2.049x | 2.072x |
| COCO Caption | 1.0023 | 1.0008 | -0.0016 | 66.63 | 64.30 | -3.49% | 1.386x | 1.337x |
| CharXiv Reasoning | 0.7413 | 0.7449 | +0.0036 | 56.56 | 54.63 | -3.42% | 1.450x | 1.400x |
| MMMU-Pro | 0.8415 | 0.8454 | +0.0040 | 59.34 | 56.53 | -4.73% | 1.508x | 1.437x |
| **Overall** | **0.8833** | **0.8832** | **-0.0000** | **61.15** | **59.66** | **-2.44%** | **1.593x** | **1.554x** |

The overall acceptance length is the unweighted mean across the five benchmarks. Overall throughput is computed as
total output tokens divided by total decode wall time, rather than as an average of per-benchmark throughput.

## Matched 10-sample comparison

| Method | Mean accept length | Overall tok/s | Overall speedup |
|---|---:|---:|---:|
| Cached autoregressive baseline | N/A | 38.40 | 1.000x |
| Original DFlash | 0.8833 | 61.15 | 1.593x |
| **DFlash + pooled MLP, 3 epochs** | **0.8832** | **59.66** | **1.554x** |

## Visual gate multiplier ablation

To isolate whether the trained visual residual is useful but under- or over-weighted, the same final checkpoint was
evaluated with its learned scalar gate multiplied by 0, 0.5, 1, and 2 at load time. All other model, decoding, sample,
and token-budget settings were held fixed. A multiplier of 0 disables the residual contribution but still executes the
MLP, so throughput differences between these rows are timing variation rather than a compute-saving bypass.

| Gate multiplier | GQA | TextVQA | COCO Caption | CharXiv | MMMU-Pro | Mean accept length | Delta vs. 0x |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0x** | **0.8083** | **1.0228** | **1.0023** | **0.7413** | **0.8415** | **0.883256** | **0.000000** |
| 0.5x | 0.7982 | 1.0228 | 1.0055 | 0.7449 | 0.8388 | 0.882038 | -0.001218 |
| 1x | 0.8070 | 1.0181 | 1.0008 | 0.7449 | 0.8454 | 0.883242 | -0.000015 |
| 2x | 0.7919 | 1.0244 | 1.0008 | 0.7402 | 0.8336 | 0.878165 | -0.005092 |

The 0x run reproduces original DFlash's acceptance length exactly on every benchmark. There is no monotonic benefit
from increasing the visual residual: 0.5x and 1x provide no aggregate gain, while 2x lowers mean acceptance length by
0.0051 (0.58%) relative to 0x. This indicates that the learned pooled-MLP residual direction has no net value on this
evaluation, rather than merely requiring a larger gate value.

Raw ablation results:

- `benchmark_results/dflash_visual_mlp_gate0_legacy5_10x256.json`
- `benchmark_results/dflash_visual_mlp_gate0p5_legacy5_10x256.json`
- `benchmark_results/dflash_visual_mlp_gate1_legacy5_10x256.json`
- `benchmark_results/dflash_visual_mlp_gate2_legacy5_10x256.json`

## Historical 4-sample results

These runs used only the first four samples of each benchmark. They are retained for context, but are not directly
mixed with the matched 10-sample result above because the additional six samples materially changed the baseline
throughput and benchmark composition.

| Method | Mean accept length | Overall tok/s | Overall speedup |
|---|---:|---:|---:|
| Original DFlash | 0.8886 | 63.51 | 1.963x |
| DFlash + Q-Former stage 1 | 0.8808 | 63.04 | 1.948x |
| DFlash + Q-Former stage 2 | 0.8582 | 59.32 | 1.833x |
| Domino | 0.8839 | 53.08 | 1.640x |
| ViSpec chain | 1.4731 | 50.42 | 1.558x |
| ViSpec tree | 2.4612 | 64.81 | 2.003x |
| DFlash + pooled MLP, 3 epochs | 0.8861 | 60.28 | 1.863x |

## Conclusion

The pooled MLP did not improve the original DFlash result in the matched 10-sample run. Its mean acceptance length is
effectively identical to original DFlash: 0.883242 versus 0.883256, a difference of -0.0017%. The additional
visual-adapter computation reduces aggregate throughput by 2.44%, from 61.15 to 59.66 tok/s, and reduces speedup over
the matched cached autoregressive baseline from 1.593x to 1.554x.

The gate multiplier ablation strengthens this conclusion: disabling the visual residual exactly recovers original
DFlash acceptance, and amplifying the residual does not reveal a hidden gain. More epochs or gate rescaling are
therefore unlikely to rescue this exact global masked-pooling + MLP + scalar-gate design. A follow-up should change the
information path itself, for example by making conditioning token- or region-aware, rather than continuing this run.

Ten samples per dataset are more stable than the previous four-sample run but remain a small benchmark. Differences of
a few thousandths in acceptance length should therefore be treated as noise-level evidence. Throughput decreased on
four of five datasets, while TextVQA improved by 1.11%; a single sequential timing run cannot separate adapter overhead
from all run-to-run variance. The aggregate result nevertheless provides no evidence that the adapter recovers its
cost through greater acceptance. ViSpec tree uses a different proposal topology, so its acceptance-rate denominator is
not directly interchangeable with the linear DFlash and ViSpec-chain denominators.
