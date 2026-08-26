#!/usr/bin/env python3
"""Redact API-key fields from an SGLang log stream."""

from __future__ import annotations

import re
import sys


PATTERNS = (
    re.compile(r"(?i)(api_key=)'[^']*'"),
    re.compile(r'(?i)("api[_-]?key"\s*:\s*)"[^"]*"'),
)

for line in sys.stdin:
    for pattern in PATTERNS:
        line = pattern.sub(r"\1'REDACTED'", line)
    sys.stdout.write(line)
    sys.stdout.flush()
