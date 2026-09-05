# Arm B load failure: `nvidia/Qwen3.8-Flash-Next-NVFP4` on 2x DGX Spark

## What was attempted

The published recipe, unchanged except for the one flag the checkpoint itself
forces:

| Setting | Arm A (lane) | Arm B (nvidia) |
| --- | --- | --- |
| `--quantization` | `modelopt_fp4` | `modelopt_mixed` |
| everything else | identical | identical |

`modelopt_mixed` is the correct flag: the pinned SGLang image registers
`"modelopt_mixed" -> ModelOptMixedPrecisionConfig`, which requires
`quant_algo == "MIXED_PRECISION"` plus a non-empty `quantized_layers` map, and
NVIDIA's `hf_quant_config.json` satisfies both. The flag was accepted and the
loader selected the right path:

```
[TP0] Using ModelOptModelLoader due to ModelOpt quantization config.
[TP0] Model is already quantized, loading directly...
[TP0] FlashInfer TRTLLM MoE deferred finalize is disabled
      (moe_runner_backend=auto, quant_method=ModelOptNvFp4FusedMoEMethod)
[TP0] using attn output gate!
```

## What happened

The process is killed by the host OOM killer during weight loading, every time,
before the API ever comes up. Rank 1 reports it plainly:

```
RuntimeError: Rank 0 scheduler died during initialization (exit code: -9).
If exit code is -9 (SIGKILL), a common cause is the OS OOM killer.
```

and rank 0 sees only the far side dropping:

```
Error ignored in is_in_the_same_node: ... Connection closed by peer [10.100.120.1]
RuntimeError: ... Connection closed by peer [10.100.120.1]:56223. This is
typically caused by a remote worker crashing.
```

With `--restart unless-stopped` this becomes a crash loop: the container reaches
"using attn output gate!" and dies roughly every 2-4 minutes.

## Attempts

| # | Change | Result |
| --- | --- | --- |
| 1 | Published `MEM_FRACTION_STATIC=0.80` | host OOM kill during load |
| 2 | `MEM_FRACTION_STATIC=0.70` (more headroom) | host OOM kill during load |

Two attempts were the budget; a third would need a changed hypothesis rather
than a repeat, so the arm was stopped and the nodes released.

## Why this is not simply "the file is too big"

It is not a raw size problem. The two checkpoints are within 2% of each other on
disk and the NVIDIA one is the *smaller* of the two in high-precision tensors:

| | lane (RadixArk) | nvidia |
| --- | ---: | ---: |
| on disk | 126,586 MiB | 126,586 MiB |
| BF16 | 16.0 GB | **11.0 GB** |
| FP8 E4M3 | 58.7 GB | 61.3 GB |
| U8 (packed FP4) | 60.4 GB | 60.4 GB |
| BF16 tensor count | 1,432 | 2,966 |

So the excess memory is being spent by the `modelopt_mixed` load path itself —
a per-layer `quantized_layers` map with 292 explicitly excluded modules, versus
the lane checkpoint's 13 wildcard excludes — not by the weights on disk. The
lane checkpoint loads and serves in the same slot, on the same two boxes, with
the same image, minutes earlier and minutes later.

## Consistency with NVIDIA's own model card

The card lists **vLLM** as the only supported runtime and **B200 / B300** as the
supported hardware. GB10 / SM121 and SGLang are both outside what NVIDIA states
it supports for this checkpoint, so this result contradicts nothing NVIDIA
claims. It only answers our question: it is not a drop-in replacement here.

## What would be needed to revisit

- A vLLM-based recipe on the Sparks rather than the SGLang lane, or
- CPU-offload / sharded loading for the excluded high-precision modules, or
- A newer SGLang build whose MIXED_PRECISION loader is less memory-hungry.

None of these are the published lane, so none were in scope for this comparison.
