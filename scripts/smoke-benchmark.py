#!/usr/bin/env python3
"""Functional and throughput checks for the Qwen SGLang endpoint."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def request_json(url: str, api_key: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def red_png_data_url(size: int = 32) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + bytes((255, 0, 0)) * size for _ in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def chat(base_url: str, api_key: str, **payload: object) -> tuple[dict, float]:
    started = time.perf_counter()
    result = request_json(f"{base_url}/chat/completions", api_key, dict(payload))
    return result, time.perf_counter() - started


def benchmark(base_url: str, api_key: str, model: str, concurrency: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a compact Python function that validates a topological "
                    "ordering. Return code only."
                ),
            }
        ],
        "max_tokens": 192,
        "temperature": 0.2,
        "reasoning_effort": "low",
    }

    def one(_: int) -> dict:
        result, elapsed = chat(base_url, api_key, **payload)
        usage = result.get("usage", {})
        return {
            "seconds": elapsed,
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(concurrency)))
    wall = time.perf_counter() - started
    tokens = sum(item["completion_tokens"] for item in results)
    return {
        "concurrency": concurrency,
        "wall_seconds": round(wall, 3),
        "completion_tokens": tokens,
        "aggregate_tps": round(tokens / wall, 2) if wall else 0,
        "mean_request_seconds": round(
            sum(item["seconds"] for item in results) / len(results), 3
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--secret-file", default=str(root / ".sglang-api-key"))
    parser.add_argument("--concurrency", default="1,4,8,16")
    args = parser.parse_args()

    api_key = Path(args.secret_file).read_text(encoding="utf-8").strip()
    models = request_json(f"{args.base_url}/models", api_key)
    print(json.dumps({"models": [item["id"] for item in models["data"]]}))

    coding, coding_seconds = chat(
        args.base_url,
        api_key,
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": "Implement binary search in Python. Return code only.",
            }
        ],
        max_tokens=192,
        temperature=0.1,
        reasoning_effort="low",
    )
    coding_message = coding["choices"][0]["message"]
    coding_tokens = int(coding.get("usage", {}).get("completion_tokens", 0))
    print(
        json.dumps(
            {
                "coding_nonempty": bool(coding_message.get("content")),
                "reasoning_present": bool(coding_message.get("reasoning_content")),
                "completion_tokens": coding_tokens,
                "seconds": round(coding_seconds, 3),
                "single_request_tps": round(coding_tokens / coding_seconds, 2),
            }
        )
    )

    tool_result, _ = chat(
        args.base_url,
        api_key,
        model=args.model,
        messages=[{"role": "user", "content": "What is the weather in Hanoi?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
        max_tokens=192,
        reasoning_effort="low",
    )
    tool_calls = tool_result["choices"][0]["message"].get("tool_calls", [])
    print(json.dumps({"tool_call_names": [x["function"]["name"] for x in tool_calls]}))

    vision_result, _ = chat(
        args.base_url,
        api_key,
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the dominant color? One word."},
                    {"type": "image_url", "image_url": {"url": red_png_data_url()}},
                ],
            }
        ],
        max_tokens=64,
        reasoning_effort="low",
    )
    print(
        json.dumps(
            {"vision": vision_result["choices"][0]["message"].get("content", "")}
        )
    )

    for concurrency in (int(value) for value in args.concurrency.split(",")):
        print(json.dumps({"benchmark": benchmark(args.base_url, api_key, args.model, concurrency)}))


if __name__ == "__main__":
    main()
