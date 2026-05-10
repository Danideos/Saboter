#!/usr/bin/env python3
"""Export metrics JSONL files as self-contained HTML dashboards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from saboter.metrics_dashboard import load_metrics_jsonl, save_metrics_dashboard


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Saboter metrics JSONL to HTML.")
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--html-out", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args(argv)

    rows = load_metrics_jsonl(args.metrics_jsonl)
    html_out = args.html_out or args.metrics_jsonl.with_suffix(".html")
    title = args.title or f"Saboter Metrics - {args.metrics_jsonl.parent.name}"
    save_metrics_dashboard(html_out, rows, title=title)
    print(f"wrote={html_out} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
