# `nvidia/Qwen3.8-Flash-Next-NVFP4` under vLLM, 2x DGX Spark TP=2 — partial run

**Date:** 2026-09-05 · **Hardware:** 2x DGX Spark (GB10, sm121, aarch64, ~120 GiB
*unified* memory each), TP=2 over the direct RoCE link · **Engine:**
`vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174`
(arm64 digest `sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b`,
reports `0.28.1rc1.dev437+ge962733e0`)

Follow-up to [`../2026-09-05-nvidia-nvfp4-vs-lane/`](../2026-09-05-nvidia-nvfp4-vs-lane/),
which found that the lane's pinned SGLang recipe cannot load this checkpoint on
this hardware at all. NVIDIA's card names vLLM as the supported engine, so this
run asks the narrower question the SGLang result could not answer: does it load
and serve under the engine NVIDIA actually supports?

## Decision: **undecided — insufficient evidence**

The lane's rule is unchanged and was fixed before any measurement:

> "Better" = no worse on quality (agreement/KL + task battery, within noise)
> **and** greater-or-equal speed. Faster with worse quality is "different
> recipe", not better.

Neither gate was reached. **No speed number, no quality number, and no KV-pool
figure exists for this arm.** The run ended when both Sparks became unreachable
during the post-weight-load stage and the nodes had to be handed back to
another lane. Anyone reading this for a checkpoint decision should read the
SGLang result as still standing: **the lane keeps
`PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark`**, because nothing here
displaces it.

What this run *does* settle is where the wall actually is, and it is not where
the SGLang run left it.

## Finding 1 — the weights load. The SGLang blocker is engine-specific.

The SGLang arm died in the OOM killer *during weight loading*, at
`MEM_FRACTION_STATIC` 0.80 and again at 0.70. Under vLLM the same checkpoint on
the same two boxes loads cleanly:

```
Model loading took 61.13 GiB memory and 407.051269 seconds
```

per node — 11/11 safetensors shards, both ranks, twice over (attempts A1 and
A2). vLLM read `hf_quant_config.json` on its own and resolved `--quantization
modelopt` to `modelopt_mixed`; the mixed-precision layout (MoE experts FP4,
attention and shared experts BF16) is not what stopped SGLang here.

That is a real negative-to-positive correction on the earlier result, and it is
the one durable thing this run produced.

## Finding 2 — the MTP heads do not load in this build

Attempt A1 asked for the checkpoint's MTP speculative decoding
(`{"method":"mtp","num_speculative_tokens":3}`). The main model loaded; the
draft loader then failed:

```
AttributeError: Layer mtp.layers.48.mlp.experts has no parameter
'w2_weight_scale_inv' for checkpoint weight
'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'
```

The MTP experts are FP8 block-scaled (`model-fp8-mtp-ple.safetensors`) and this
build's modelopt-mixed loader has no mapping for their block-scale tensors.
**Speculative decoding is therefore unavailable for this checkpoint on this
engine today** — which matters for any future comparison, because the RadixArk
arm's measured speed *includes* MTP acceptance of ~3 tokens per verify pass. A
future nvidia-arm sweep without MTP is not speed-comparable to that arm without
saying so out loud.

## Finding 3 — the pressure moved past weight load, and 0.85 is the wrong setting for unified memory

Attempt A2 (no MTP) loaded weights on both nodes and then went quiet after:

```
Setting attention block size to 1568 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 0.13% to ensure that mamba page size and attention page size are exactly equal.
Reducing Torch threads from 20 to 1 for serving.
```

Roughly fifteen minutes later both nodes stopped answering SSH; port 8890 never
opened; node 1 later stopped answering ICMP as well. That is the host thrashing
itself off the network, not a clean engine error, so there is no traceback to
quote — the honest statement is that **the process ran out of host memory during
post-load profiling / graph capture**, and the evidence for it is circumstantial
(both boxes, same stage, no engine-side error line).

The likely cause is a configuration error of ours rather than a property of the
checkpoint: GB10's ~120 GiB is **unified**, host and device drawing on one pool.
`--gpu-memory-utilization 0.85` reserves 85% of that pool for the engine while
the host-side runtime is still spending from the same pool — behaviour that does
not bite on the discrete B200/B300 that NVIDIA tested. On a 61.13 GiB weight
load per node, 0.85 leaves very little for everything else.

**The next attempt should start at `--gpu-memory-utilization` 0.70-0.75** before
anything else is varied, and should watch host free memory through the
profiling stage rather than only the GPU pool.

## Cost / config table

| | value |
| --- | ---: |
| Image (arm64) | `vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174` |
| vLLM at boot | `0.28.1rc1.dev437+ge962733e0` |
| Required floor from NVIDIA's card | `d4d703ca` (2026-09-03) — image is 76 commits ahead |
| Weight load, per node | 61.13 GiB / 407.05 s |
| Shard load, cold cache | 11 shards, 6 m 43 s, 37.6 s/shard |
| Quantization resolved to | `modelopt_mixed` |
| MTP | fails to load (see Finding 2) |
| Multimodal | disabled (`--limit-mm-per-prompt {"image":0,"video":0}`), text-only |
| Boot to first `/v1/models` | **never reached** |
| KV pool | **not measured** |

No 3x3 chart accompanies this result, because there is no sweep to chart. The
harness that would have produced one is committed here anyway
([`../../scripts/vllm-context-sweep.py`](../../scripts/vllm-context-sweep.py)),
so the next attempt starts at the boot, not at the tooling.

## Engine selection (reproducible)

Three candidates were checked against the commit floor on NVIDIA's card,
`d4d703caf908786416585ceb1f369e2e0363358b` ("[Bugfix][Model] Fix FP8 PLE loading
in mixed ModelOpt checkpoints", merged 2026-09-03 17:09 UTC — two days before
this run):

| candidate | vLLM version | vs. floor | verdict |
| --- | --- | --- | --- |
| `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (already on the nodes) | `0.25.2.dev0+g752a3a504` | diverged, 2343 behind | no |
| `vllm/vllm-openai:nightly-65b7662d3…` (already on the nodes) | `0.26.1rc1.dev602+g65b7662d3` | 1024 behind | no |
| `vllm/vllm-openai:nightly-e962733e0…` | `0.28.1rc1.dev437+ge962733e0` | 76 ahead | **selected** |

Full detail, including the GB10 smoke test and the multi-node mechanism, in
[`engine-selection.json`](engine-selection.json). Two points worth carrying
forward:

- **`torch.cuda.get_arch_list()` is not a go/no-go signal on GB10.** The
  selected image lists `sm_80 … sm_120` with no `sm_121`, and runs on GB10
  anyway (`get_device_capability() == (12, 1)`, bf16 matmul fine) — CUDA family
  compatibility covers it.
- **Ray is not installed in the upstream image and is not needed.** Multi-node
  TP works through native `--nnodes / --node-rank / --master-addr /
  --master-port`. The follower node **must** get `--headless`, or its EngineCore
  aborts with `AssertionError: collective_rpc should not be called on follower
  node`.

## Files

- [`engine-selection.json`](engine-selection.json) — the three candidates, the commit comparison, the GB10 smoke test, the multi-node mechanism
- [`boot-attempts.json`](boot-attempts.json) — every attempt, its flags, its decisive log lines, and what was not measured
- [`../../scripts/vllm-start-node.sh`](../../scripts/vllm-start-node.sh) — the two-node launcher used here (addresses parameterised)
- [`../../scripts/vllm-context-sweep.py`](../../scripts/vllm-context-sweep.py) — the lane sweep ported to vLLM (metrics namespace, prefix-cache reset endpoint, optional API key); unused so far

## Honest limits

- Finding 3's cause is inferred, not logged. The boxes died before writing a
  reason. Treat "host OOM at post-load" as the leading hypothesis, not a proven
  fact.
- Two boots is a small sample, and both used `--gpu-memory-utilization 0.85`.
  The setting was never varied, so nothing here shows the checkpoint *cannot*
  serve on 2x GB10 — only that it did not at 0.85.
- The comparison this branch was opened to make (speed, agreement, task battery
  against the RadixArk arm) is not started. The RadixArk arm's own task battery
  is still lost; `out-lane-radixark-nvfp4-tasks.json` in the scratch tree is a
  file of `Connection refused` errors, so an engine-matched re-run of that arm is
  still owed regardless of what happens to this one.
