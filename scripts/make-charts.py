#!/usr/bin/env python3
"""3x3 sweep charts per checkpoint, plus a delta chart.

Panels (the lane's chart shape):
  1 prompt processing (tok/s)      2 generation (tok/s)        3 cold TTFT (log)
  4 warm TTFT                      5 ms/token                  6 accepted tokens/pass
  7 prompt KB/s                    8 generation B/s            9 cold / warm TTFT

Usage:
  make_charts.py single <sweep.json> <out.png> "<title>"
  make_charts.py delta  <armA.json> <armB.json> <out.png> "<title>"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("prompt_tok_per_s", "Prompt processing", "tokens/s", False),
    ("gen_tok_per_s", "Generation", "tokens/s", False),
    ("cold_ttft_s", "Cold TTFT", "seconds", True),
    ("warm_ttft_s", "Warm TTFT", "seconds", False),
    ("ms_per_token", "Per-token latency", "ms/token", False),
    ("accepted_tokens_per_pass", "Accepted tokens per pass", "tokens/verify", False),
    ("prompt_kb_per_s", "Prompt throughput", "KB/s", False),
    ("gen_bytes_per_s", "Generation throughput", "bytes/s", False),
    ("cold_warm_ttft_ratio", "Cold / warm TTFT", "ratio", False),
]
ARMS = [("thinking_off", "thinking off", "#1f77b4", "o"),
        ("thinking_on", "thinking on", "#d62728", "s")]


def series(doc: dict, arm: str, field: str):
    xs, ys = [], []
    for r in doc.get("results", []):
        a = (r.get("arms") or {}).get(arm)
        if not a or a.get("error"):
            continue
        v = a.get(field) if field in a else (a.get("median") or {}).get(field)
        pt = (a.get("median") or {}).get("prompt_tokens") or r.get("target_tokens")
        if v is not None and pt:
            xs.append(pt)
            ys.append(v)
    return xs, ys


def grid(title: str):
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    fig.suptitle(title, fontsize=15, y=0.995)
    return fig, axes.ravel()


def finish(fig, out: str):
    fig.tight_layout(rect=[0, 0.01, 1, 0.975])
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def single(path: str, out: str, title: str) -> None:
    doc = json.loads(Path(path).read_text())
    fig, axes = grid(title)
    for ax, (field, name, unit, logy) in zip(axes, PANELS):
        for arm, label, color, marker in ARMS:
            xs, ys = series(doc, arm, field)
            if xs:
                ax.plot(xs, ys, marker=marker, color=color, label=label,
                        linewidth=1.6, markersize=4)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("prompt tokens")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    finish(fig, out)


def delta(a_path: str, b_path: str, out: str, title: str) -> None:
    A = json.loads(Path(a_path).read_text())
    B = json.loads(Path(b_path).read_text())
    fig, axes = grid(title)
    for ax, (field, name, unit, _logy) in zip(axes, PANELS):
        for arm, label, color, marker in ARMS:
            xa, ya = series(A, arm, field)
            xb, yb = series(B, arm, field)
            m = {}
            for x, y in zip(xa, ya):
                m[round(x / 1000)] = y
            xs, ys = [], []
            for x, y in zip(xb, yb):
                k = round(x / 1000)
                if k in m and m[k]:
                    xs.append(x)
                    ys.append((y - m[k]) / m[k] * 100.0)
            if xs:
                ax.plot(xs, ys, marker=marker, color=color, label=label,
                        linewidth=1.6, markersize=4)
        ax.axhline(0, color="#444", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("prompt tokens")
        ax.set_ylabel("% change (nvidia vs lane)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    finish(fig, out)


if __name__ == "__main__":
    if sys.argv[1] == "single":
        single(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        delta(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
