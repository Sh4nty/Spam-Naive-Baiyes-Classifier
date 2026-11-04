"""Reproduce all local, data-backed project outputs in one command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = [
        ["scripts/run_baseline.py"],
        ["scripts/run_attacks.py"],
        ["scripts/run_m5.py"],
        ["scripts/evaluate_a7.py"],
        ["scripts/run_extended_experiments.py"],
    ]
    for command in commands:
        print(f"==> {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, *command], cwd=root, check=True)
    print("All local experiments completed.")
    print("Fresh-100 and Reuters are optional external-corpus evaluations; "
          "see docs/dataset_datasheet.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
