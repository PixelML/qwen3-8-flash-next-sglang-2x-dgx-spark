#!/usr/bin/env python3
"""List served models without placing the API key in process arguments."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    secret_file = Path(os.environ.get("SGLANG_SECRET_FILE", root / ".sglang-api-key"))
    api_key = secret_file.read_text(encoding="utf-8").strip()
    port = os.environ.get("API_PORT", "8888")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        print(json.dumps(json.load(response), separators=(",", ":")))


if __name__ == "__main__":
    main()
