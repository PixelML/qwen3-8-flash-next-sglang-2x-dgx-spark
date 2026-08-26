# Qwen3.8-Flash-Next on 2× DGX Spark

Reproducible SGLang deployment of the 176B-parameter, 6B-active
`Qwen3.8-Flash-Next` preview across two NVIDIA DGX Spark systems.

This recipe uses an NVFP4 checkpoint, tensor parallelism across both GB10 GPUs,
and direct RoCE networking. It also documents two SM121 compatibility changes
needed by the pinned day-zero SGLang image.

## Verified configuration

| Component | Pin |
| --- | --- |
| Model | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Model revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
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

The pinned SGLang image routes Qwen Sparse Attention decode to FlashInfer only
on SM100. On GB10/SM121 it falls back to a FlashAttention-4 CuTe kernel that
fails during compilation. FlashInfer supports the same decode API through XQA
on SM120/SM121, so [`patches/sglang-qsa-sm121-xqa.patch`](patches/sglang-qsa-sm121-xqa.patch)
widens that capability check. The startup script extracts the original source,
applies the two-line patch, and bind-mounts the result read-only.

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
- `patches/` — minimal SGLang SM121 compatibility patch.
- `results/` — measured throughput and validation evidence.
- `docs/TROUBLESHOOTING.md` — known failure signatures and fixes.

## License

MIT. The referenced models and container image retain their own licenses.
