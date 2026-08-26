#!/usr/bin/env python3
"""Verify that every safetensors shard referenced by the model index exists."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <model-directory>")

    model_dir = Path(sys.argv[1])
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise SystemExit("checkpoint index or config.json is missing")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise SystemExit("checkpoint index contains no shard references")

    missing = [name for name in shards if not (model_dir / name).is_file()]
    empty = [name for name in shards if (model_dir / name).is_file() and not (model_dir / name).stat().st_size]
    if missing or empty:
        raise SystemExit(f"checkpoint incomplete: missing={missing}, empty={empty}")

    total = sum((model_dir / name).stat().st_size for name in shards)
    print(f"verified {len(shards)} shards ({total:,} bytes)")


if __name__ == "__main__":
    main()
