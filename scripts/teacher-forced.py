#!/usr/bin/env python3
"""Teacher-forced next-token distributions over a fixed text set.

Uses /v1/completions with echo=true, logprobs=K, max_tokens=0, so the server
returns the model's next-token distribution at every position of a FIXED token
sequence. Because the sequence is fixed and both checkpoints share a
byte-identical tokenizer (verified), the two arms are scored at exactly the same
positions and the comparison is teacher-forced, not free-running.

Records per position: the argmax token and the top-K (token -> logprob) map.
Comparison is done offline by compare_quality.py.

Honest limit: the API exposes only the top-K entries, so the KL reported by
compare_quality.py is computed over the union of the two arms' top-K supports,
renormalized. It is not a full-vocabulary KL. That is stated in the result.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("Q_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("Q_MODEL", "qwen3.8-flash-next")
SECRET_FILE = os.environ.get("Q_SECRET", "/opt/qwen38-sglang/.sglang-api-key")
IN_PATH = os.environ.get("Q_TF_IN", "data/agreement_set.jsonl")
OUT_PATH = os.environ.get("Q_TF_OUT", "tf.jsonl")
TOPK = int(os.environ.get("Q_TOPK", "20"))

API_KEY = Path(SECRET_FILE).read_text(encoding="utf-8").strip()
HDRS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def score(text: str) -> dict:
    """Teacher-forced next-token distributions over a fixed text.

    Uses SGLang's native /generate rather than /v1/completions: this build
    rejects the OpenAI echo+logprobs combination outright with
    "Echo is not compatible with logprobs. To compute logprobs of input prompt,
    please use the native /generate API." /generate returns
    input_top_logprobs as [[logprob, token_id, token_text], ...] per position.
    """
    body = {
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": TOPK,
    }
    req = urllib.request.Request(f"{BASE}/generate",
                                 data=json.dumps(body).encode(),
                                 headers=HDRS, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    meta = data.get("meta_info") or {}
    in_lp = meta.get("input_token_logprobs") or []
    in_top = meta.get("input_top_logprobs") or []

    tokens, token_logprobs, top_logprobs = [], [], []
    for i, entry in enumerate(in_lp):
        lp = entry[0] if isinstance(entry, (list, tuple)) else None
        tid = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
        tokens.append(str(tid))
        token_logprobs.append(lp)
        top = {}
        if i < len(in_top) and in_top[i]:
            for cand in in_top[i]:
                if isinstance(cand, (list, tuple)) and len(cand) >= 2:
                    top[str(cand[1])] = cand[0]
        top_logprobs.append(top)
    return {"tokens": tokens, "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs}


def main() -> None:
    items = [json.loads(l) for l in Path(IN_PATH).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(OUT_PATH).open("w", encoding="utf-8")
    t0 = time.time()
    for i, it in enumerate(items):
        try:
            rec = score(it["text"])
            rec["id"] = it["id"]
            rec["bucket"] = it["bucket"]
            rec["ok"] = True
        except Exception as exc:  # noqa: BLE001
            rec = {"id": it["id"], "bucket": it["bucket"], "ok": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(items)} ({round(time.time()-t0,1)}s)", flush=True)
    out.close()
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
