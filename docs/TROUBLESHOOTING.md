# Troubleshooting

## FlashAttention-4 CuTe `weakly congruent` failure

**Symptom:** rank startup fails while compiling Qwen Sparse Attention decode,
with a CuTe message saying the view coordinates and shape are weakly
congruent.

**Cause:** the pinned SGLang image only selects FlashInfer's TRT-LLM decode API
on SM100. GB10 reports SM121, so SGLang falls back to a FlashAttention-4 CuTe
path that does not compile for this shape.

**Fix:** run `scripts/apply-sm121-qsa-patch.sh`. It extracts the exact original
file from the pinned image and applies `patches/sglang-qsa-sm121-xqa.patch`.
The patched source uses FlashInfer XQA on SM120/SM121.

## Linear-attention state dtype failure

**Symptom:** CUDA-graph capture fails because the FlashInfer prefill kernel
rejects BF16 state, or decode fails because state is not BF16.

**Fix:** keep the phase split used by this repository:

```text
--linear-attn-prefill-backend triton
--linear-attn-decode-backend flashinfer
--mamba-ssm-dtype bfloat16
```

## Rank 1 waits indefinitely

**Checks:**

1. `RANK0_ADDR` must be the rank-zero address on the direct RoCE subnet.
2. `DIST_PORT` must be open on that direct interface.
3. Both nodes must use the same `NCCL_SOCKET_IFNAME`, HCA, and GID index.
4. Confirm `ibv_devinfo` and an ordinary IP ping work over the direct link.
5. Start rank 1 before rank 0, as `scripts/start-cluster.sh` does.

## API is healthy but single-request TPS is low

This deployment's non-speculative baseline is approximately 25–26 output
tok/s for one stream, while aggregate throughput reaches roughly 200–250 tok/s
under continuous batching. A public TLS proxy can reduce the observed
single-request figure further.

The official SGLang low-latency profile enables NEXTN/MTP with three steps and
four draft tokens. Treat it as a separate tuning profile on SM121: benchmark
acceptance rate, memory headroom, output quality, and rank stability before
using it for production.

## TLS hostname fails only through a proxy

Check the configured hostname for an accidental trailing dot before the port.
Some TLS clients treat `host.:8443` differently from `host:8443`, producing a
misleading decode or certificate error even though the direct API is healthy.
