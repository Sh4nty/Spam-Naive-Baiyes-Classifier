"""Evaluate a locally supplied binary corpus without fabricating missing data.

Example:
    python scripts/evaluate_optional_corpus.py path/to/fresh100.csv

The file must contain text, label and timestamp columns (custom names are
available as command-line options). The victim is trained only on Enron
training data and the supplied corpus is evaluated as an external test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_nb.data import standardize
from spam_nb.experiment import MODELS, RESULTS, load_config, prepare_data
from spam_nb.metrics import bootstrap_ci, classification_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--timestamp-col", default="timestamp")
    args = parser.parse_args()

    import pandas as pd

    cfg = load_config()
    raw = pd.read_csv(args.corpus)
    corpus = standardize(
        raw, args.text_col, args.label_col, args.timestamp_col
    )
    train = prepare_data(cfg)["enron_train"]

    results = {}
    for name, cls in MODELS.items():
        model = cls(alpha=cfg["smoothing_alpha"]).fit(train.text, train.label)
        predictions = model.predict(corpus.text)
        results[name] = {
            "metrics": classification_metrics(corpus.label, predictions),
            "ci": bootstrap_ci(
                corpus.label, predictions, cfg["bootstrap_resamples"],
                seed=cfg["seed"],
            ),
            "rows": len(corpus),
        }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"{args.corpus.stem}_evaluation.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
