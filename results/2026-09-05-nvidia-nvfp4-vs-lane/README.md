# nvidia/Qwen3.8-Flash-Next-NVFP4 vs the lane checkpoint — 2x DGX Spark

**Date:** 2026-09-05 · **Hardware:** 2x NVIDIA DGX Spark (GB10, SM121, ~120 GiB
unified each), TP=2 over direct RoCE · **Engine:** the lane's pinned
`lmsysorg/sglang@sha256:12d3392b…` (`0.0.0.dev1+gd91c3682b`) with this repo's
SM121 QSA + token-0 guard patches.

## Decision: **not better — it does not run here**

Stated before any measurement was taken:

> "Better" = no worse on quality (agreement/KL + task battery, within noise)
> **and** greater-or-equal speed. Faster with worse quality is "different
> recipe", not better.

`nvidia/Qwen3.8-Flash-Next-NVFP4` never reached the speed or quality gates,
because it **cannot be loaded by this recipe on this hardware**. It is killed by
the host OOM killer during weight loading, at the published
`MEM_FRACTION_STATIC=0.80` and again at `0.70`. Full evidence:
[`armB-load-failure.md`](armB-load-failure.md).

**The lane keeps `PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark`.** No change
to the published recipe's checkpoint is warranted.

This is a negative result about *this recipe on this hardware*, not about the
checkpoint's quality. NVIDIA's card lists vLLM as the supported runtime and
B200/B300 as the supported hardware; GB10 and SGLang are outside both. Nothing
here contradicts NVIDIA's own claims.

## Table 1 — Cost

| | lane (RadixArk NVFP4) | nvidia (MIXED_PRECISION) |
| --- | ---: | ---: |
| Hub size | 135.3 GB (419 files) | 132.7 GB (25 files) |
| On disk | 126,586 MiB | 126,586 MiB (132,734,514,674 B) |
| BF16 / FP8 / packed-FP4 | 16.0 / 58.7 / 60.4 GB | 11.0 / 61.3 / 60.4 GB |
| `quant_algo` | `NVFP4` (uniform) | `MIXED_PRECISION` (experts only) |
| Excluded modules | 13 globs | 292 explicit |
| SGLang flag | `modelopt_fp4` | `modelopt_mixed` |
| Boot to first `/v1/models` | **574 s** | never booted |
| Free device mem after load | 21.4 GB per GB10 | n/a |
| KV pool at 262,144 ctx | 568,704 tokens | n/a |
| License | Qwen Community 1.0 | NVIDIA Open Model + Qwen Community 1.0 |

Both need two Sparks: at ~126 GiB against ~120 GiB per GB10, neither fits on one.

## Table 2 — Speed (lane checkpoint, measured)

Ten prompt lengths, thinking off/on alternated per length, cold + warm + 4 reps
each, 512 output tokens. Token counts come from the server's own final SSE usage
object; acceptance is the exact delta of SGLang's cumulative
`generation_tokens_total` / `spec_verify_calls_total` counters bracketing each
sample, not the periodic gauge. Medians over the 4 reps.

| Prompt tok | Mode | Cold TTFT s | Warm TTFT s | Cold/warm | Gen tok/s | ms/token | Accepted/pass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 324 | off | 0.362 | 0.391 | 0.93 | 50.31 | 19.95 | 3.03 |
| 343 | on | 0.252 | 0.163 | 1.55 | 55.12 | 18.17 | 3.26 |
| 855 | off | 0.605 | 0.159 | 3.80 | 50.57 | 19.78 | 3.00 |
| 829 | on | 0.624 | 0.610 | 1.02 | 41.66 | 26.08 | 2.60 |
| 1,918 | off | 1.223 | 0.140 | 8.71 | 45.64 | 21.91 | 2.79 |
| 1,975 | on | 0.724 | 0.166 | 4.36 | 62.17 | 16.11 | 3.62 |
| 3,808 | off | 1.868 | 0.177 | 10.57 | 48.24 | 20.74 | 3.00 |
| 3,838 | on | 2.254 | 0.167 | 13.49 | 44.44 | 22.57 | 2.74 |
| 8,162 | off | 3.978 | 0.175 | 22.68 | 47.04 | 21.35 | 3.07 |
| 8,203 | on | 2.755 | 0.144 | 19.12 | 41.59 | 24.05 | 2.51 |
| 16,306 | off | 5.294 | 0.239 | 22.14 | 45.12 | 22.16 | 2.86 |
| 16,338 | on | 9.073 | 0.154 | 59.10 | 42.33 | 23.78 | 2.69 |
| 33,561 | off | 13.837 | 0.265 | 52.32 | 51.07 | 19.59 | 3.05 |
| 33,639 | on | 11.031 | 0.255 | 43.33 | 40.56 | 24.66 | 2.43 |
| 67,078 | off | 22.625 | 0.350 | 64.65 | 46.21 | 21.64 | 2.75 |
| 67,159 | on | 25.566 | 0.365 | 69.96 | 42.96 | 23.28 | 2.57 |
| 133,176 | off | 47.366 | 0.676 | 70.05 | 46.52 | 21.55 | 2.85 |
| 133,306 | on | 51.206 | 0.571 | 89.65 | 43.09 | 23.28 | 2.60 |
| 255,645 | off | 106.755 | 0.973 | 109.76 | 43.69 | 22.90 | 2.88 |
| **258,054** | **on** | 99.626 | **163.376** | **0.61** | **7.62** | **176.11** | **1.00** |

Chart: [`armA-3x3.png`](armA-3x3.png)

### The one place this recipe falls over

The last row is not noise. At ~258k prompt tokens **with thinking on**:

- acceptance collapses to exactly **1.00** — speculative decoding stops
  contributing anything, every drafted token is rejected;
- generation drops to **7.6 tok/s**, a 5.7x slowdown against the same length
  with thinking off (43.7 tok/s);
- warm TTFT (163 s) exceeds cold TTFT (100 s), i.e. the prefix cache stops
  helping and starts hurting;
- the SM121 token-0 guard fires: `dspark: token-id-0 loop after 17 output tokens`
  followed by `resetting prefix cache before the next prefill`.

Everything below 258k, and 258k with thinking off, is healthy and flat: 40-62
tok/s across a 780x range of prompt length. The failure is specific to maximum
context combined with reasoning.

**After that guard fires the engine wedges**: HTTP keeps answering, the scheduler
stops. Two independent occurrences, each costing a ~10 minute restart. That is
worth knowing for anyone running this recipe at full context.

## Table 3 — Quality

**Not measured as a comparison, because there is only one arm.** What was built
and what it produced:

| | Status |
| --- | --- |
| BF16 reference (`Qwen/Qwen3.8-Flash-Next`) | **Impossible here.** 360 GB against 240 GiB total across both Sparks. Established before spending GPU time; no download attempted. |
| Teacher-forced agreement + KL, lane arm | **Done**: 200 fixed texts, 39,009 scored positions, top-20 per position ([`tf-summary-lane.json`](tf-summary-lane.json)) |
| Teacher-forced, nvidia arm | Not possible — never loaded |
| GSM8K 50 / HumanEval 20 | **Not obtained on either arm.** The engine wedged during the concurrency sweep that preceded it; every battery request returned connection-refused. |

The lane arm's teacher-forced baseline is kept because it is the fixed reference
any future second arm can be scored against, at zero extra cost:

| Bucket | Items | Scored positions | Mean forced-token logprob | Mean top-1 logprob | Mean top-20 entropy (nats) |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 50 | 7,864 | -0.216 | -0.068 | 0.133 |
| math | 50 | 8,743 | -0.617 | -0.181 | 0.374 |
| prose | 50 | 11,636 | -1.406 | -0.376 | 0.718 |
| json | 50 | 10,766 | -1.408 | -0.821 | 1.096 |
| **all** | **200** | **39,009** | **-0.988** | **-0.393** | **0.627** |

Because no BF16 anchor is reachable on this hardware, even a completed two-arm
comparison would have shown only *that* the checkpoints differ, not which is
closer to the unquantized model. That limit was stated in the design before the
run, not discovered afterwards.

## Engine stability findings (independent of the comparison)

Two things about the pinned image that the lane's own users should know:

1. **`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` defaults to `True`.** Under
   sustained mixed-length load the KV/mamba page accounting drifts by a few
   thousand tokens out of ~570k (the engine logs `mamba num: 58` while
   `#running-req` is 1) and the strict check turns that drift into a fatal
   `ValueError: pool memory leak detected!`. The engine died four times in the
   first 40 minutes, each costing a ~10 minute reload. Setting it to `0` demotes
   it to a warning and the sweep then ran for hours. This branch adds a
   `STRICT_MEM_CHECK_DURING_IDLE` knob that **defaults to 1**, preserving
   published behaviour for anyone who does not opt in.
2. **`ignore_eos` / `min_tokens` trip the same check immediately.** They were
   dropped from the harness; generation length is elicited by the prompt instead.
3. **Concurrency ≥ 4 at 4k prompts with 512-token outputs killed the engine**
   (c=1 measured 42.1 tok/s; c=4 and c=8 both took it down). The README's
   published concurrency figures used 192-token requests, a much lighter profile.

## Reproducing

```bash
# lane arm
QUANTIZATION=modelopt_fp4 STRICT_MEM_CHECK_DURING_IDLE=0 ./scripts/start-cluster.sh
Q_ARM=lane Q_OUT=sweep.json python3 scripts/context-sweep.py

# nvidia arm (fails to load; kept for reproduction)
MODEL_REPO=nvidia/Qwen3.8-Flash-Next-NVFP4 \
MODEL_REVISION=fab0aecb760cec45227f6656abcaafa11abca87a ./scripts/prepare-model.sh
QUANTIZATION=modelopt_mixed ./scripts/start-cluster.sh
```

## Files

- [`armA-sweep.json`](armA-sweep.json) — every sample, all 10 lengths, both modes
- [`armA-3x3.png`](armA-3x3.png) — the 3x3 chart
- [`tf-summary-lane.json`](tf-summary-lane.json) — teacher-forced baseline
- [`armA-conc.json`](armA-conc.json) — concurrency, including the c>=4 failures
- [`checkpoint-comparison.json`](checkpoint-comparison.json) — both checkpoints' metadata
- [`armB-load-failure.md`](armB-load-failure.md) — the OOM evidence

## Attribution

- **NVIDIA** — [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), quantized with Model Optimizer v0.46.0, NVIDIA Open Model License.
- **RadixArk** — [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), the community NVFP4 conversion this lane serves.
- **Qwen** — [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen Community License 1.0.
- **MiaAI-Lab** — the SM121 QSA fallback and token-0 guard this recipe ports.
