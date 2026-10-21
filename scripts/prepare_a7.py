"""Build the A7 input sample: 300 defanged spam messages that both victims
flag when clean.

    python scripts/prepare_a7.py [--n 300]

Writes data/a7/a7_sample.jsonl (tracked in git, so teammates can run the
LLM generation step without the full data pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_nb.experiment import MODELS, ROOT, fit_victim, load_config, prepare_data  # noqa: E402
from spam_nb.llm_camouflage import has_live_contact, sample_spam  # noqa: E402

OUT = ROOT / "data" / "a7" / "a7_sample.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    cfg = load_config()
    data = prepare_data(cfg)
    models = [fit_victim(name, data, cfg) for name in MODELS]
    sample = sample_spam(data["enron_test"], models, n=args.n, seed=cfg["seed"])
    assert not any(has_live_contact(t) for t in sample.original), "defanging failed"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for row in sample.itertuples(index=False):
            fh.write(json.dumps({"id": row.id, "original": row.original}) + "\n")
    print(f"Wrote {len(sample)} messages to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
