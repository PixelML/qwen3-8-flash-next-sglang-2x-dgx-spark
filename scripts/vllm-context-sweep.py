#!/usr/bin/env python3
"""Qwen3.8-Flash-Next 2x DGX Spark prompt-length sweep, thinking on vs off (vLLM).

vLLM port of scripts/context-sweep.py. Same prompts, same ladder, same cold/warm
protocol; only the server-specific bits differ (metrics namespace, prefix-cache
reset endpoint, optional API key).

One boot per checkpoint, arms alternated per length.

Ported from ~/WIP/cmp170hx-knowledge/.scratch/context_sweep.py (the 3x3 chart's
source), which was itself ported from this lane's scripts/prefill-benchmark.py.
Adapted from vLLM/GLM to SGLang/Qwen3.8:

  - thinking toggled via chat_template_kwargs.enable_thinking (verified in the
    checkpoint's chat_template.jinja lines 46/165).
  - acceptance read from SGLang's own cumulative Prometheus counters
    sglang:generation_tokens_total and sglang:spec_verify_calls_total, bracketing
    each sample window. Accepted tokens per verify pass = d(gen tokens)/d(verify
    calls). These are Counters, so the delta is exact for the window; the
    sglang:spec_accept_length Gauge is a periodic average and is NOT used.
  - token counts come from the final SSE usage object
    (stream_options.include_usage), never counted from stream events.

Per (length, arm): a cold sample (unique nonce payload -> guaranteed prefix-cache
miss), then an immediate identical repeat (warm, prefix-cache hit), then REPS
further unique samples. Arms alternate per length so thermal/interference drift
is shared across arms rather than confounded with arm order.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = os.environ.get("Q_URL", "http://127.0.0.1:8890")
MODEL = os.environ.get("Q_MODEL", "qwen3.8-flash-next")
OUT_PATH = os.environ.get("Q_OUT", "sweep.json")
ARM_LABEL = os.environ.get("Q_ARM", "unknown")
SECRET_FILE = os.environ.get("Q_SECRET", "/opt/qwen38-sglang/.sglang-api-key")
OUT_TOKENS = int(os.environ.get("Q_OUT_TOKENS", "512"))
REPS = int(os.environ.get("Q_REPS", "4"))
CONTEXT_LIMIT = int(os.environ.get("Q_CONTEXT_LIMIT", "262144"))
# A request aborted by the token-0 guard leaves the HTTP stream open, so an
# over-generous client timeout turns one guard abort into an hour of dead
# wall-clock. Sized well above the worst legitimate sample (a ~107 s cold
# prefill at 253k plus a 512-token decode).
REQ_TIMEOUT_S = int(os.environ.get("Q_REQ_TIMEOUT_S", "600"))
LENGTHS = [int(x) for x in os.environ.get(
    "Q_LENGTHS", "327,860,2000,4000,8000,16000,33000,66000,131000,258000").split(",")]

_secret = Path(SECRET_FILE)
API_KEY = _secret.read_text(encoding="utf-8").strip() if _secret.is_file() else ""
HDRS = {"Content-Type": "application/json"}
if API_KEY:
    HDRS["Authorization"] = f"Bearer {API_KEY}"

WORDS = (
    "amber", "birch", "cobalt", "delta", "ember", "falcon", "granite",
    "harbor", "indigo", "juniper", "kelp", "linen", "meadow", "nickel",
    "opal", "prairie",
)


def build_prompt(word_count: int, seed: int) -> str:
    rng = random.Random(seed)
    payload = " ".join(rng.choice(WORDS) for _ in range(word_count))
    return (f"Unique sample {seed}. Read this payload:\n{payload}\n"
            "Summarize this payload, then explain in detail how you would "
            "verify that your summary is faithful to it.")


def read_metrics() -> dict:
    """Cumulative vLLM counters. Returns {} if /metrics is unavailable.

    vLLM's names differ from SGLang's: generation/prompt token counters carry the
    `vllm:` prefix, and speculative acceptance is exposed as accepted-vs-drafted
    token counters rather than verify-pass counts. Accepted tokens per pass is
    therefore d(accepted)/d(drafts) when spec decoding is on, and None when it is
    off (this arm: the checkpoint's MTP heads do not load, see the result README).
    """
    keys = ("generation_tokens_total", "prompt_tokens_total",
            "spec_decode_num_accepted_tokens_total",
            "spec_decode_num_drafts_total",
            "spec_decode_num_draft_tokens_total")
    out = {k: 0.0 for k in keys}
    try:
        req = urllib.request.Request(f"{BASE}/metrics", headers=HDRS)
        with urllib.request.urlopen(req, timeout=20) as r:
            for raw in r:
                line = raw.decode(errors="replace")
                if line.startswith("#"):
                    continue
                for key in keys:
                    if line.startswith(f"vllm:{key}"):
                        try:
                            out[key] += float(line.rsplit(" ", 1)[1])
                        except (ValueError, IndexError):
                            pass
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def spec_delta(before: dict, after: dict) -> dict:
    d_gen = after.get("generation_tokens_total", 0) - before.get("generation_tokens_total", 0)
    d_acc = (after.get("spec_decode_num_accepted_tokens_total", 0)
             - before.get("spec_decode_num_accepted_tokens_total", 0))
    d_drafts = (after.get("spec_decode_num_drafts_total", 0)
                - before.get("spec_decode_num_drafts_total", 0))
    return {
        "gen_tokens_delta": round(d_gen, 2),
        "verify_calls_delta": round(d_drafts, 2),
        "accepted_tokens_per_pass": (
            round((d_acc + d_drafts) / d_drafts, 4) if d_drafts > 0 else None),
    }


def flush_cache() -> bool:
    """Release cached radix + mamba state between blocks.

    On this build the linear-attention (mamba) state pool is retained alongside
    the radix cache. Across a long sweep of unique long prompts it fills up
    (observed: "mamba num: 58" while #running-req is 1) and the scheduler's
    invariant checker eventually raises "pool memory leak detected" and takes the
    engine down. Flushing between (length, arm) blocks bounds that growth. It is
    done at block boundaries only, never between a cold sample and its warm
    repeat, so the cold/warm prefix-cache contrast stays intact.
    """
    try:
        req = urllib.request.Request(f"{BASE}/reset_prefix_cache", headers=HDRS, method="POST")
        with urllib.request.urlopen(req, timeout=120):
            return True
    except Exception:  # noqa: BLE001
        return False


def wait_for_server(max_wait_s: int = 1500) -> bool:
    """Block until the API answers again after an engine restart.

    The container runs with --restart unless-stopped, so a scheduler crash is
    followed by a ~9-10 minute reload. Samples taken during that window would be
    recorded as failures and would silently hollow out the sweep, so instead the
    harness waits and re-takes the sample.
    """
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{BASE}/v1/models", headers=HDRS)
            with urllib.request.urlopen(req, timeout=15) as r:
                if r.status == 200:
                    time.sleep(5)
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(15)
    return False


def is_conn_error(rec: dict) -> bool:
    e = (rec or {}).get("error") or ""
    return any(k in e for k in ("URLError", "RemoteDisconnected", "ConnectionResetError",
                                "IncompleteRead", "BadStatusLine", "Connection refused"))


def chat_sample(word_count: int, seed: int, thinking: bool) -> dict:
    prompt = build_prompt(word_count, seed)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUT_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        # NOTE: ignore_eos / min_tokens are deliberately NOT sent. On this pinned
        # SGLang build they reproducibly trip the scheduler KV invariant checker
        # ("pool memory leak detected", leaked_full_pages) and kill the engine.
        # Length is elicited by the instruction and capped by max_tokens; the
        # ACTUAL completion_tokens is recorded per sample so any length
        # asymmetry between the two checkpoints stays visible.
    }
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=HDRS, method="POST")
    started = time.perf_counter()
    first_at = None
    last_at = started
    usage = {}
    gen_chars = 0
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT_S) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                usage = ev.get("usage") or usage
                for ch in ev.get("choices", []):
                    delta = ch.get("delta") or {}
                    piece = ""
                    for field in ("content", "reasoning", "reasoning_content"):
                        val = delta.get(field)
                        if val:
                            piece += val
                    if piece:
                        if first_at is None:
                            first_at = time.perf_counter()
                        gen_chars += len(piece)
                        last_at = time.perf_counter()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "seed": seed, "thinking": thinking}

    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    if first_at is None or pt <= 0:
        return {"ok": False, "error": "no first token or prompt usage",
                "seed": seed, "thinking": thinking}

    ttft = first_at - started
    decode_s = max(last_at - first_at, 1e-9)
    total_s = last_at - started
    prompt_bytes = len(prompt.encode("utf-8"))
    gen_bytes = gen_chars  # ascii payloads; recorded as observed stream bytes
    return {
        "ok": True,
        "seed": seed,
        "thinking": thinking,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": round(ttft, 5),
        "decode_s": round(decode_s, 5),
        "total_s": round(total_s, 5),
        "prompt_tok_per_s": round(pt / ttft, 3) if ttft > 0 else None,
        "prompt_kb_per_s": round((prompt_bytes / 1024.0) / ttft, 3) if ttft > 0 else None,
        "gen_tok_per_s": round((ct - 1) / decode_s, 3) if ct > 1 else None,
        "ms_per_token": round(decode_s * 1000.0 / (ct - 1), 4) if ct > 1 else None,
        "gen_bytes_per_s": round(gen_bytes / decode_s, 3),
        "prompt_bytes": prompt_bytes,
        "gen_chars": gen_chars,
    }


def chat_sample_resilient(word_count: int, seed: int, thinking: bool) -> dict:
    """One sample, retried once across an engine restart."""
    rec = chat_sample(word_count, seed, thinking)
    if rec.get("ok") or not is_conn_error(rec):
        return rec
    print("    engine unreachable; waiting for restart", flush=True)
    if not wait_for_server():
        rec["waited_for_restart"] = "timed out"
        return rec
    flush_cache()
    rec2 = chat_sample(word_count, seed, thinking)
    rec2["retried_after_engine_restart"] = True
    return rec2


def _probe_prompt_tokens(word_count: int, seed: int, thinking: bool):
    """One max_tokens=1 probe; returns the server's own prompt_tokens count."""
    probe = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(word_count, seed)}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(probe).encode(),
                                 headers=HDRS, method="POST")
    try:
        pt = 0
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT_S) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    ev = json.loads(line[6:])
                    if ev.get("usage"):
                        pt = int(ev["usage"].get("prompt_tokens", 0))
        return pt or None
    except Exception:  # noqa: BLE001
        return None


def calibrate_word_count(target_tokens: int, seed: int, thinking: bool) -> int:
    """Pick a word count whose prompt lands within 5% of target_tokens.

    Ratio-based rather than a blind binary search. One cheap probe measures this
    payload's actual tokens-per-word, then the count is scaled and refined. The
    old binary search needed up to 9 probes and, at the top of the ladder, each
    probe is a full-length prefill -- at 258k it both cost ~13 minutes per arm and
    still converged badly, overshooting to 322k tokens, which the server truncated
    to the 262,144 context and left 0 tokens for output.

    A hard ceiling keeps prompt + reserved output inside the context window, so a
    calibration miss can never silently turn into a truncated, unmeasurable sample.
    """
    ceiling = CONTEXT_LIMIT - OUT_TOKENS - 512  # template + safety headroom
    target = min(target_tokens, ceiling)

    seed_words = max(8, min(400, target // 4))
    pt = _probe_prompt_tokens(seed_words, seed, thinking)
    if not pt:
        return max(1, int(target * 0.72))
    overhead = 40  # chat template, roughly constant
    per_word = max(0.2, (pt - overhead) / seed_words)

    wc = max(1, int((target - overhead) / per_word))
    for _ in range(3):
        got = _probe_prompt_tokens(wc, seed, thinking)
        if not got:
            return wc
        if got > ceiling:
            wc = max(1, int(wc * (ceiling / got) * 0.97))
            continue
        if abs(got - target) <= 0.05 * target:
            return wc
        wc = max(1, int(wc * (target / got)))
    return wc


def run_length(target_tokens: int, run_seed: int) -> dict:
    entry = {"target_tokens": target_tokens, "arms": {}}
    for thinking in (False, True):
        arm = "thinking_on" if thinking else "thinking_off"
        seed_base = run_seed + (7919 if thinking else 0)
        if not wait_for_server():
            entry["arms"][arm] = {"error": "engine did not come back"}
            continue
        flush_cache()
        wc = calibrate_word_count(target_tokens, seed_base, thinking)

        m0 = read_metrics()
        cold = chat_sample_resilient(wc, seed_base + 1000, thinking)
        m1 = read_metrics()
        cold.update({f"spec_{k}": v for k, v in spec_delta(m0, m1).items()})

        # warm: identical prompt immediately after -> prefix cache hit
        warm = chat_sample_resilient(wc, seed_base + 1000, thinking)
        m2 = read_metrics()
        warm.update({f"spec_{k}": v for k, v in spec_delta(m1, m2).items()})

        reps = []
        prev = m2
        for i in range(REPS):
            s = chat_sample_resilient(wc, seed_base + 2000 + i * 37, thinking)
            m = read_metrics()
            s.update({f"spec_{k}": v for k, v in spec_delta(prev, m).items()})
            prev = m
            reps.append(s)

        ok_reps = [r for r in reps if r.get("ok")]

        def med(field):
            vals = [r[field] for r in ok_reps if r.get(field) is not None]
            return round(statistics.median(vals), 4) if vals else None

        entry["arms"][arm] = {
            "word_count": wc,
            "cold": cold,
            "warm": warm,
            "reps": reps,
            "median": {
                "prompt_tokens": med("prompt_tokens"),
                "completion_tokens": med("completion_tokens"),
                "ttft_s": med("ttft_s"),
                "prompt_tok_per_s": med("prompt_tok_per_s"),
                "prompt_kb_per_s": med("prompt_kb_per_s"),
                "gen_tok_per_s": med("gen_tok_per_s"),
                "ms_per_token": med("ms_per_token"),
                "gen_bytes_per_s": med("gen_bytes_per_s"),
                "accepted_tokens_per_pass": med("spec_accepted_tokens_per_pass"),
            },
            "cold_ttft_s": cold.get("ttft_s"),
            "warm_ttft_s": warm.get("ttft_s"),
            "cold_warm_ttft_ratio": (
                round(cold["ttft_s"] / warm["ttft_s"], 4)
                if cold.get("ok") and warm.get("ok") and warm.get("ttft_s") else None
            ),
        }
        print(f"  [{arm}] wc={wc} cold_ttft={cold.get('ttft_s')} "
              f"warm_ttft={warm.get('ttft_s')} gen={entry['arms'][arm]['median']['gen_tok_per_s']} "
              f"acc={entry['arms'][arm]['median']['accepted_tokens_per_pass']}", flush=True)
    return entry


def main() -> None:
    meta = {
        "arm_label": ARM_LABEL,
        "model": MODEL,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_tokens": OUT_TOKENS,
        "reps": REPS,
        "lengths_requested": LENGTHS,
        "results": [],
    }
    for length in LENGTHS:
        print(f"[length {length}]", flush=True)
        t0 = time.time()
        try:
            meta["results"].append(run_length(length, run_seed=length * 13 + 5))
        except Exception as exc:  # noqa: BLE001
            meta["results"].append({"target_tokens": length, "error": str(exc)})
            print(f"  ERROR {exc}", flush=True)
        print(f"  [length {length}] done in {round(time.time()-t0,1)}s", flush=True)
        Path(OUT_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(OUT_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
