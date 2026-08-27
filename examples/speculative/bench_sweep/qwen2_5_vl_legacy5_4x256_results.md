# Qwen2.5-VL speculative decoding: legacy five-benchmark results

This report records a small inference-throughput comparison run on 2026-08-03.
It compares cached autoregressive decoding, DFlash, Domino, and ViSpec with and
without its candidate tree. This is a performance benchmark, not a task-accuracy
evaluation.

## Common benchmark parameters

| Parameter | Value |
|---|---|
| Target | Qwen2.5-VL-7B-Instruct |
| Inference engine | Local Transformers reference evaluator |
| GPU | NVIDIA B200 |
| Target dtype | BF16 |
| Attention implementation | SDPA |
| Batch size | 1 |
| Decoding | Greedy |
| Benchmarks | GQA, TextVQA, COCO Caption, CharXiv Reasoning, MMMU-Pro |
| Samples per benchmark | 4 |
| Output per sample | Exactly 256 tokens; EOS does not stop generation |
| System prompt | None |
| Dataset config | [`vlm_spec_bench_datasets_legacy_256.yaml`](vlm_spec_bench_datasets_legacy_256.yaml) |

All methods used the same prompts, output limits, target checkpoint, and
Transformers inference path. The aggregate throughput is calculated as total
output tokens divided by the sum of the five measured generation times.

## Method parameters

| Method | Draft checkpoint | Draft parameters | Proposal and verification |
|---|---|---|---|
| Baseline | None | Cached target autoregressive decoding | One target token per step |
| DFlash | `checkpoints/epoch_6_step_14574/model/consolidated` | 3 draft layers; block size 8; target layers 1, 13, 25 | Parallel block proposal; block verification |
| Domino | `domino_checkpoints/epoch_6_step_14574/model/consolidated` | Same DFlash backbone; GRU hidden size 1024; projection dim 256; pure-draft prefix 1; shifted labels | Block size 8 plus causal Domino correction; block verification |
| ViSpec, no tree | `/data/models/ViSpec-Qwen2.5-VL-7B-Instruct-nemo` | 1 draft layer; 2 visual query tokens; recursive depth 3 | Top-k 1 chain; proposal token cap 5 including root |
| ViSpec, tree | `/data/models/ViSpec-Qwen2.5-VL-7B-Instruct-nemo` | 1 draft layer; 2 visual query tokens; recursive depth 3 | Top-k 8 tree; proposal token cap 30 including root |

## Aggregate results

Acceptance length below includes the one target-guaranteed token emitted in
every verification round.

| Method | Throughput (tok/s) | Speedup vs baseline | Acceptance length, including 1 |
|---|---:|---:|---:|
| Baseline | 32.36 | 1.000x | 1.000 |
| DFlash | 63.51 | 1.963x | 1.884 |
| Domino | 53.08 | 1.640x | 1.848 |
| ViSpec, no tree | 50.42 | 1.558x | 2.422 |
| ViSpec, tree | **64.81** | **2.003x** | **3.427** |

## Speedup by benchmark

| Method | GQA | TextVQA | COCO Caption | CharXiv Reasoning | MMMU-Pro | Aggregate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x |
| DFlash | 2.218x | **2.732x** | 1.368x | 2.004x | 1.444x | 1.963x |
| Domino | 1.575x | 2.649x | 1.098x | 1.854x | 1.193x | 1.640x |
| ViSpec, no tree | 2.056x | 1.571x | 1.112x | 1.672x | 1.353x | 1.558x |
| ViSpec, tree | **2.570x** | 2.297x | **1.467x** | **2.008x** | **1.563x** | **2.003x** |

## Acceptance length by benchmark

Every value includes the target-guaranteed `1`.

| Method | GQA | TextVQA | COCO Caption | CharXiv Reasoning | MMMU-Pro | Aggregate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DFlash | 1.784 | 2.004 | 1.921 | 1.783 | 1.951 | 1.884 |
| Domino | 1.476 | 2.264 | 1.854 | 1.975 | 1.852 | 1.848 |
| ViSpec, no tree | 3.151 | 2.073 | 2.212 | 2.560 | 2.370 | 2.422 |
| ViSpec, tree | **4.112** | **3.075** | **3.357** | **3.531** | **3.230** | **3.427** |

The aggregate acceptance length is computed across verification rounds, not as
an unweighted average of the five displayed benchmark values.

## Throughput by benchmark

| Method | GQA | TextVQA | COCO Caption | CharXiv Reasoning | MMMU-Pro |
|---|---:|---:|---:|---:|---:|
| Baseline | 27.14 | 23.77 | 49.04 | 30.05 | 45.59 |
| DFlash | 60.20 | **64.95** | 67.07 | 60.21 | 65.81 |
| Domino | 42.74 | 62.98 | 53.85 | 55.70 | 54.37 |
| ViSpec, no tree | 55.81 | 37.34 | 54.51 | 50.25 | 61.66 |
| ViSpec, tree | **69.77** | 54.60 | **71.96** | **60.32** | **71.26** |

Values are output tokens per second. Because this run contains only four
samples per benchmark, use it as a fast comparison rather than a final
statistical performance claim.

## Reproduction

Use the common command below and substitute the method-specific arguments from
the following table:

```bash
.venv/bin/python tools/transformers_vlm_spec_bench.py \
  --target /data/models/Qwen2.5-VL-7B-Instruct \
  --draft <DRAFT> \
  --mode <MODE> \
  --benchmark-suite legacy \
  --num-prompts 4 \
  --fixed-output-length \
  --attn-implementation sdpa \
  <METHOD_ARGS> \
  --output <OUTPUT>
```

| Method | `<DRAFT>` | `<MODE>` | `<METHOD_ARGS>` |
|---|---|---|---|
| Baseline | Target checkpoint | `baseline` | None |
| DFlash | DFlash checkpoint above | `dflash` | `--verification-mode block --baseline-results benchmark_results/baseline_legacy5_4x256.json` |
| Domino | Domino checkpoint above | `dflash` | `--verification-mode block --baseline-results benchmark_results/baseline_legacy5_4x256.json` |
| ViSpec, no tree | ViSpec checkpoint above | `vispec` | `--vispec-proposal-mode chain --baseline-results benchmark_results/baseline_legacy5_4x256.json` |
| ViSpec, tree | ViSpec checkpoint above | `vispec` | `--vispec-proposal-mode tree --baseline-results benchmark_results/baseline_legacy5_4x256.json` |

## Raw results

- [Baseline JSON](../../../benchmark_results/baseline_legacy5_4x256.json)
- [DFlash JSON](../../../benchmark_results/dflash_legacy5_4x256.json)
- [Domino JSON](../../../benchmark_results/domino_legacy5_4x256.json)
- [ViSpec no-tree JSON](../../../benchmark_results/vispec_chain_legacy5_4x256.json)
- [ViSpec tree JSON](../../../benchmark_results/vispec_tree_legacy5_4x256.json)

## 10-sample Domino follow-up

On 2026-08-19, Domino was rerun three times with 10 samples per benchmark and the same fixed 256-token output,
Transformers/SDPA engine, block size 8, greedy proposal, target checkpoint, and cached baseline as the existing
10-sample DFlash run. Domino throughput below is the median of the three aggregate runs; acceptance is deterministic
across the repeats. Acceptance length includes the one target-guaranteed token.

| Method | Samples | Mean acceptance length | Aggregate tok/s | Speedup vs. cached baseline |
|---|---:|---:|---:|---:|
| Cached target baseline | 50 | 1.0000 | 38.40 | 1.000x |
| DFlash | 50 | 1.8833 | 61.15 | 1.593x |
| **Domino, three-run median** | **50** | **1.9431** | **59.63** | **1.553x** |

Domino's three aggregate throughput runs were 59.63, 61.70, and 56.66 tok/s. Relative to DFlash, Domino increased
mean acceptance length by 0.0599 (3.18%); its median throughput point estimate is 2.50% below the single DFlash run.
The acceptance improvement is deterministic across the three repeats. The throughput difference is not yet decisive:
Domino's observed range spans both sides of the DFlash result, and DFlash has not yet received matching repeats.

| Domino repeat | Completed samples | Output tokens | Mean acceptance length | Aggregate tok/s |
|---|---:|---:|---:|---:|
| Run 1, with BF16 parity diagnostics | 50 | 12,800 | 1.9431 | 59.63 |
| Run 2 | 50 | 12,800 | 1.9431 | 61.70 |
| Run 3 | 50 | 12,800 | 1.9431 | 56.66 |
| **Median** | **50** | **12,800** | **1.9431** | **59.63** |

| Benchmark | DFlash accept | Domino accept | DFlash tok/s | Domino median tok/s | Throughput change |
|---|---:|---:|---:|---:|---:|
| GQA | 1.8083 | 1.6106 | 58.32 | 51.76 | -11.25% |
| TextVQA | 2.0228 | 2.2528 | 66.35 | 69.43 | +4.65% |
| COCO Caption | 2.0023 | 1.9884 | 66.63 | 63.76 | -4.30% |
| CharXiv Reasoning | 1.7413 | 1.9155 | 56.56 | 60.42 | +6.82% |
| MMMU-Pro | 1.8415 | 1.9483 | 59.34 | 56.72 | -4.41% |

The first Domino repeat also recorded BF16 token-parity diagnostics against an independently generated cached-greedy
target reference. Those diagnostics are retained in the raw JSON but are excluded from this comparison by experiment
policy. PyTorch emitted its existing non-contiguous-GRU warning during each Domino process, so these values describe
the current reference runtime rather than an optimized fused Domino implementation.

Raw 10-sample results:

- [Cached baseline](../../../benchmark_results/baseline_legacy5_10x256.json)
- [DFlash](../../../benchmark_results/dflash_legacy5_10x256.json)
- [Domino run 1 with parity diagnostics](../../../benchmark_results/domino_legacy5_10x256_run1_parity.json)
- [Domino run 2](../../../benchmark_results/domino_legacy5_10x256_run2.json)
- [Domino run 3](../../../benchmark_results/domino_legacy5_10x256_run3.json)
