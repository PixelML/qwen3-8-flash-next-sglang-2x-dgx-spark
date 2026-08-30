# Troubleshooting

## FlashAttention-4 CuTe `weakly congruent` failure

**Symptom:** rank startup fails while compiling Qwen Sparse Attention decode,
with a CuTe message saying the view coordinates and shape are weakly
congruent.

**Cause:** the pinned SGLang image only selects FlashInfer's TRT-LLM decode API
on SM100. GB10 reports SM121, so SGLang falls back to a FlashAttention-4 CuTe
path that does not compile for this shape.

**Fix:** run `scripts/apply-sm121-qsa-patch.sh`. It extracts the exact original
file from the pinned image and applies `patches/sglang-qsa-sm121-fallback.patch`.
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

The non-speculative baseline is approximately 25–26 output tok/s for one
stream. The verified NEXTN/MTP profile reaches 47.54 tok/s on the repeatable
single-stream test and 53.86 tok/s on a separate coding response.

Confirm all four speculative flags from `scripts/start-node.sh` are present in
the container command. Startup logs should show draft-weight loading and MTP
CUDA-graph capture. Runtime logs should report accepted draft lengths and
rates; this test observed acceptance rates around 0.6–0.8. A public TLS proxy
can still reduce client-observed throughput.

With MTP, the configured request limit of 36 becomes an effective limit of 25
because each request needs additional Mamba state. This is expected; lowering
context length or changing memory allocation may change the limit but requires
a new quality, memory, and stability test.

## TLS hostname fails only through a proxy

Check the configured hostname for an accidental trailing dot before the port.
Some TLS clients treat `host.:8443` differently from `host:8443`, producing a
misleading decode or certificate error even though the direct API is healthy.
