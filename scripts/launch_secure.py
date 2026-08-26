"""Launch SGLang while loading its API key from a mounted secret file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def main() -> None:
    load_plugins()
    secret_path = Path(
        os.environ.get("SGLANG_API_KEY_FILE", "/run/secrets/sglang-api-key")
    )
    api_key = secret_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(f"SGLang API key file is empty: {secret_path}")

    # ServerArgs becomes immutable after resolution. Add the key to the
    # in-memory argument list before parsing so it never appears in host argv.
    server_args = prepare_server_args([*sys.argv[1:], "--api-key", api_key])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
