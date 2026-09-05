# Qwen3.8-Flash-Next on 2× DGX Spark

Reproducible SGLang deployment of the 176B-parameter, 6B-active
`Qwen3.8-Flash-Next` preview across two NVIDIA DGX Spark systems.

This recipe uses an NVFP4 checkpoint, tensor parallelism across both GB10 GPUs,
and direct RoCE networking. It also documents two SM121 compatibility changes
needed by the pinned day-zero SGLang image.

The exact validated weights are available from the attributed PixelML mirror:
**[`PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark`](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark)**.
The model card preserves the original Qwen license and credits RadixArk for the
NVFP4 conversion; PixelML did not modify or rebrand the quantized tensors.

## Verified configuration

| Component | Pin |
| --- | --- |
| Model | [`PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark`](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark) |
| Model revision | `b80180e371f13348ec49641a6e66999e7854b179` |
| Original checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Source model | `Qwen/Qwen3.8-Flash-Next` |
| Source revision | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| SGLang image | `lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1` |
| SGLang version | `0.0.0.dev1+gd91c3682b` |
| Parallelism | TP=2, two nodes, one GB10 GPU per node |
| Context | 262,144 tokens |
| Quantization | ModelOpt NVFP4, FlashInfer CUTLASS GEMM/MoE |
| Speculative decoding | NEXTN/MTP: 3 steps, top-k 1, 4 draft tokens |

The deployment follows the
[official SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
and adapts it to the memory and SM121 kernel constraints of DGX Spark.

## Measured performance

Direct SGLang measurements with NEXTN/MTP and 192-token coding requests:

| Concurrent requests | Aggregate output tok/s |
| ---: | ---: |
| 1 | 47.54 |
| 4 | 87.55 |
| 8 | 158.17 |
| 16 | 275.37 |

NEXTN/MTP improved the repeatable single-stream result by 82% over the
non-speculative 26.09 tok/s baseline. A separate coding response reached 53.86
tok/s. The scheduler accepts 25 simultaneous requests with this profile even
though `MAX_RUNNING_REQUESTS=36`, because MTP allocates additional Mamba state.
See
[`results/RESULTS-2026-08-26.md`](results/RESULTS-2026-08-26.md) for methodology
and the before/after results.

### Uncached prefill

Client-observed prompt throughput with one forced output token, a unique prompt
prefix per request, and three fully warm samples per size:

| Target prompt | Median actual prompt | Median TTFT | Median input tok/s |
| ---: | ---: | ---: | ---: |
| 1K | 1,046 | 0.4524 s | 2,327.78 |
| 4K | 4,103 | 1.4924 s | 2,758.65 |
| 16K | 16,471 | 5.5729 s | **2,960.12** |

The input rate includes HTTP, tokenization, scheduling, and the first generated
token, making it a conservative end-to-end prefill number. Server logs reported
zero cached prompt tokens for all 50 prefill batches in the final run window.
See [`results/PREFILL-2026-08-27.md`](results/PREFILL-2026-08-27.md) for raw
samples and run:

```bash
python3 scripts/prefill-benchmark.py
```

### Checkpoint scoreboard

| Checkpoint | Quant scheme | Serves on this recipe | Verdict |
| --- | --- | --- | --- |
| [`PixelML/…-Dual-DGX-Spark`](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark) (RadixArk) | uniform `NVFP4`, `modelopt_fp4` | **yes**, boot 574 s | **in use** |
| [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | `MIXED_PRECISION`, `modelopt_mixed` | **no** — host OOM during weight load at `MEM_FRACTION_STATIC` 0.80 and 0.70 | not a drop-in here |

NVIDIA's official NVFP4 build was benchmarked head-to-head on 2026-09-05 and
could not be served by this recipe on GB10; its model card lists vLLM and
B200/B300, so this is outside what NVIDIA supports for it. Full evidence,
including the lane checkpoint's complete 327 -> 258k prompt-length sweep, is in
[`results/2026-09-05-nvidia-nvfp4-vs-lane/`](results/2026-09-05-nvidia-nvfp4-vs-lane/README.md).

### Known limits at full context

At ~258k prompt tokens **with thinking enabled**, speculative acceptance
collapses to 1.00 (nothing accepted), generation falls to 7.6 tok/s against 43.7
tok/s at the same length with thinking off, warm TTFT exceeds cold TTFT, and the
SM121 token-0 guard fires. After the guard fires the engine can wedge: HTTP keeps
answering while the scheduler stops. Everything below 258k is flat at 40-62
tok/s. See the result above for the full table.

Long unattended runs should also set `STRICT_MEM_CHECK_DURING_IDLE=0` (see
`scripts/start-node.sh`); the pinned image otherwise turns a few-thousand-token
KV/mamba accounting drift into a fatal `pool memory leak detected` roughly every
10-15 minutes of sustained mixed-length load.

## Requirements

- Two DGX Spark systems on the same NVIDIA system-software release.
- A direct Ethernet/RoCE link between the systems.
- The same Linux username, UID, and GID on both nodes.
- Docker configured for the non-root deployment user.
- SSH aliases from the controller to both nodes.
- At least 160 GB free storage per node for the checkpoint and caches.

## Deploy

1. Clone this repository to the same path on both Spark nodes and the controller.

2. Create the configuration on the controller, then copy it to both nodes:

   ```bash
   cp .env.example .env
   $EDITOR .env
   scp .env spark-1:/opt/qwen38-sglang/.env
   scp .env spark-2:/opt/qwen38-sglang/.env
   ```

3. Download and verify the exact checkpoint on each Spark:

   ```bash
   ./scripts/prepare-model.sh
   ```

4. Generate one API secret on the controller and copy that same file to both
   Spark nodes:

   ```bash
   umask 077
   openssl rand -hex 32 > .sglang-api-key
   scp .sglang-api-key spark-1:/opt/qwen38-sglang/.sglang-api-key
   scp .sglang-api-key spark-2:/opt/qwen38-sglang/.sglang-api-key
   ```

5. From the controller, launch both ranks and run the smoke benchmark:

   ```bash
   ./scripts/start-cluster.sh
   ssh spark-1 'cd /opt/qwen38-sglang && python3 scripts/smoke-benchmark.py'
   ```

The rank-zero node exposes an OpenAI-compatible API at
`http://<rank-zero-address>:8888/v1`. Do not expose this plaintext endpoint to
the public internet; place an authenticated TLS proxy in front of it.

## Why the SM121 patch exists

Upstream MiaAI-Lab traced long-context corruption on GB10/SM121 to
FlashInfer's TRT-LLM sparse-decode kernel: on SM121 it silently emits token
id 0 (`!`) at long context, filling a request's output budget with `!`
(sglang#36806, sglang#36537). The fix replaces that kernel with a Triton
packed-varlen fallback for the exact QSA call contract (sglang#36845), and
this repo ports both changes as unified patches that
`scripts/apply-sm121-qsa-patch.sh` applies to the sources extracted from the
stock pinned image and stages into `build/`. `scripts/start-node.sh`
bind-mounts four patched files read-only over the image's copies:

- `sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
  ([`patches/sglang-qsa-sm121-fallback.patch`](patches/sglang-qsa-sm121-fallback.patch))
  — forbids TRT-LLM sparse decode on SM121 and resolves the Triton fallback
  from `_resolve_flash_attn_varlen_func`, leaving SM100/SM120 paths intact.
- `sglang/srt/managers/schedule_batch.py`
  ([`patches/sglang-token0-guard-schedule_batch.patch`](patches/sglang-token0-guard-schedule_batch.patch))
  — aborts a request once its last 16 output samples are all token id 0.
- `sglang/srt/managers/scheduler_components/batch_result_processor.py`
  ([`patches/sglang-token0-guard-batch-result-processor.patch`](patches/sglang-token0-guard-batch-result-processor.patch))
  — keeps an aborted token-0 completion out of the radix tree.
- `sglang/srt/managers/scheduler.py`
  ([`patches/sglang-token0-guard-scheduler.patch`](patches/sglang-token0-guard-scheduler.patch))
  — resets the prefix cache before the next prefill after an abort.

The fallback kernel itself is staged as a new module,
`sglang/srt/layers/attention/qsa/sm121_varlen.py`
([`patches/sm121_varlen.py`](patches/sm121_varlen.py)), also bind-mounted
read-only.

If a request still decodes 16 consecutive token-id-0 samples, the guard
finishes it with HTTP 500 and `finish_reason=abort`; the completion is not
inserted into the radix tree, and the prefix cache is reset before the next
prefill so a later request cannot reuse the poisoned KV. This port comes
from MiaAI-Lab's DSpark work, commit `0f95001`.

Gated DeltaNet uses Triton for prefill and FlashInfer for decode with BF16
Mamba state. This split avoids the incompatible state-dtype requirements seen
when using FlashInfer for both phases on SM121.

## Why NEXTN/MTP matters

Active parameters approximate arithmetic per token. They do not include all
latency from QSA/Gated DeltaNet kernels, expert routing, memory movement, and
cross-node synchronization. The official low-latency NEXTN/MTP profile drafts
multiple tokens and verifies them together, amortizing that latency when the
drafts are accepted. On this deployment, observed acceptance rates were about
0.6–0.8 and single-stream throughput increased from 26.09 to 47.54 tok/s.

## Repository layout

- `scripts/` — model preparation, secure launch, lifecycle, and tests.
- `scripts/prefill-benchmark.py` — calibrated unique-prefix TTFT/prefill test.
- `patches/` — minimal SGLang SM121 compatibility patch.
- `results/` — measured throughput and validation evidence.
- `docs/TROUBLESHOOTING.md` — known failure signatures and fixes.

## License and model provenance

The deployment code in this repository is MIT. The weights are governed by
the
[`Qwen Community License 1.0`](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark/blob/main/LICENSE),
including separate commercial-license conditions for Model-as-a-Service and
AI Work Assistant businesses. The NVFP4 conversion is credited to
[`RadixArk`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4); the
PixelML Hugging Face repository is an exact community mirror paired with this
dual-DGX-Spark recipe.
