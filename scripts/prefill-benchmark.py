#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path


WORDS = (
    "amber",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "falcon",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "linen",
    "meadow",
    "nickel",
    "opal",
    "prairie",
)


def build_prompt(word_count: int, seed: int) -> str:
    """Builds a unique deterministic prompt for an uncached prefill sample.

    Args:
        word_count: Number of payload words to generate.
        seed: Random seed that changes the prompt from its first payload token.

    Returns:
        Prompt text containing the generated payload and a short instruction.
    """
    rng = random.Random(seed)
    payload = " ".join(rng.choice(WORDS) for _ in range(word_count))
    return f"Unique sample {seed}. Read this payload:\n{payload}\nReply OK."


def measure_prefill(
    base_url: str,
    api_key: str,
    model: str,
    word_count: int,
    seed: int,
) -> dict[str, float | int]:
    """Measures client-observed time to the first generated token.

    The prompt-token count comes from the server's usage response. The request
    forces one output token, so prompt tokens divided by TTFT is a conservative
    end-to-end prefill rate that includes HTTP, tokenization, and scheduling.

    Args:
        base_url: OpenAI-compatible API base ending in ``/v1``.
        api_key: API bearer credential.
        model: Served model name.
        word_count: Number of randomized payload words.
        seed: Unique deterministic prompt seed.

    Returns:
        Prompt-token count, TTFT, and client-observed prefill throughput.

    Raises:
        RuntimeError: If the stream omits its first token or usage accounting.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": build_prompt(word_count, seed)}
        ],
        "max_tokens": 1,
        "temperature": 0,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if first_token_at is None and any(
                    delta.get(field)
                    for field in ("content", "reasoning", "reasoning_content")
                ):
                    first_token_at = time.perf_counter()

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    if first_token_at is None or prompt_tokens <= 0:
        raise RuntimeError("stream did not return a first token and prompt usage")
    ttft = first_token_at - started
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_seconds": round(ttft, 4),
        "client_prefill_tps": round(prompt_tokens / ttft, 2),
    }


def calibrate_word_count(
    base_url: str,
    api_key: str,
    model: str,
    target_tokens: int,
    seed: int,
) -> int:
    """Calibrates payload words to within roughly five percent of a target.

    Args:
        base_url: OpenAI-compatible API base ending in ``/v1``.
        api_key: API bearer credential.
        model: Served model name.
        target_tokens: Desired API-reported prompt-token count.
        seed: First seed used for calibration requests.

    Returns:
        Calibrated payload word count.
    """
    word_count = target_tokens
    for attempt in range(2):
        sample = measure_prefill(
            base_url,
            api_key,
            model,
            word_count,
            seed=seed + attempt,
        )
        actual = int(sample["prompt_tokens"])
        if abs(actual - target_tokens) / target_tokens <= 0.05:
            break
        word_count = max(1, round(word_count * target_tokens / actual))
    return word_count


def summarize(target_tokens: int, samples: list[dict[str, float | int]]) -> dict:
    """Summarizes repeated prefill samples with medians and ranges.

    Args:
        target_tokens: Desired prompt size used to calibrate the samples.
        samples: Repeated prefill measurement records.

    Returns:
        Median prompt size, TTFT, prefill rate, and observed TPS range.
    """
    prompt_tokens = [int(item["prompt_tokens"]) for item in samples]
    ttfts = [float(item["ttft_seconds"]) for item in samples]
    rates = [float(item["client_prefill_tps"]) for item in samples]
    return {
        "target_prompt_tokens": target_tokens,
        "median_prompt_tokens": int(statistics.median(prompt_tokens)),
        "median_ttft_seconds": round(statistics.median(ttfts), 4),
        "median_client_prefill_tps": round(statistics.median(rates), 2),
        "client_prefill_tps_range": [round(min(rates), 2), round(max(rates), 2)],
    }


def main() -> None:
    """Runs calibrated, unique-prefix prefill measurements."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--secret-file", default=str(root / ".sglang-api-key"))
    parser.add_argument("--sizes", default="1024,4096,16384")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    api_key = Path(args.secret_file).read_text(encoding="utf-8").strip()
    run_seed = args.seed if args.seed is not None else time.time_ns()
    print(json.dumps({"run_seed": run_seed, "radix_cache_policy": "unique-prefix"}))
    measure_prefill(args.base_url, api_key, args.model, 256, seed=run_seed)

    for target_tokens in (int(value) for value in args.sizes.split(",")):
        size_seed = run_seed + target_tokens * 10
        word_count = calibrate_word_count(
            args.base_url,
            api_key,
            args.model,
            target_tokens,
            seed=size_seed,
        )
        measure_prefill(
            args.base_url,
            api_key,
            args.model,
            word_count,
            seed=size_seed + 100,
        )
        samples = [
            measure_prefill(
                args.base_url,
                api_key,
                args.model,
                word_count,
                seed=size_seed + 1000 + sample_index,
            )
            for sample_index in range(args.samples)
        ]
        print(json.dumps({"samples": samples}))
        print(json.dumps({"summary": summarize(target_tokens, samples)}))


if __name__ == "__main__":
    main()
