"""Clean (no-attack) baselines for both NB variants.

    python scripts/run_baseline.py

Trains Multinomial and Bernoulli NB on the Enron training split and
reports Precision, Recall, F1, accuracy and false-positive rate (spam is
the positive class) with 95% bootstrap CIs on the Enron test split and the
external SpamAssassin set. Writes results/baseline.md and .json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_nb.experiment import (  # noqa: E402
    MODELS,
    RESULTS,
    describe,
    fit_victim,
    load_config,
    prepare_data,
)
from spam_nb.metrics import evaluate  # noqa: E402

METRICS = ["precision", "recall", "f1", "accuracy", "fpr"]


def fmt(value, ci):
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main() -> int:
    cfg = load_config()
    start = time.time()
    data = prepare_data(cfg)
    print(f"Data ready in {time.time() - start:.0f}s")
    sets = {"Enron test": data["enron_test"], "SpamAssassin (external)": data["sa_test"]}

    rows, results = [], {}
    for name in MODELS:
        model = fit_victim(name, data, cfg)
        for set_name, df in sets.items():
            res = evaluate(
                model, df["text"], df["label"],
                n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"],
            )
            clean = res["clean"]
            results[f"{name}/{set_name}"] = {
                "metrics": clean["metrics"],
                "ci95": clean["ci"],
                "confusion": clean["confusion"],
                "vocabulary_size": len(model.vocabulary_),
            }
            rows.append(
                f"| {name} | {set_name} | "
                + " | ".join(fmt(clean["metrics"][m], clean["ci"][m]) for m in METRICS)
                + f" | {clean['confusion']['tp']}/{clean['confusion']['fp']}/"
                f"{clean['confusion']['tn']}/{clean['confusion']['fn']} |"
            )
            print(f"{name:12s} {set_name:24s} F1={clean['metrics']['f1']:.3f} "
                  f"P={clean['metrics']['precision']:.3f} R={clean['metrics']['recall']:.3f}")

    dd = data["dedup"]
    lines = [
        "# Clean baselines (no attack)",
        "",
        "Spam is the positive class. Values are point estimates with 95% "
        f"percentile bootstrap CIs ({cfg['bootstrap_resamples']} resamples, "
        f"seed {cfg['seed']}). Smoothing alpha = {cfg['smoothing_alpha']}, full vocabulary.",
        "",
        "## Data",
        "",
        f"- Enron train: {describe(data['enron_train'])}",
        f"- Enron test: {describe(data['enron_test'])}",
        f"- SpamAssassin (external): {describe(data['sa_test'])}",
        "",
        f"Split: {cfg['split_note']}",
        "",
        f"Near-duplicates removed (5-word shingles, Jaccard >= {dd['threshold']}): "
        f"{dd['enron_test_removed']} Enron test messages that duplicate Enron "
        f"training messages, and {dd['sa_removed']} SpamAssassin messages "
        f"({dd['sa_removed_spam']} of them spam) that duplicate Enron training "
        "messages. Enron-Spam's spam was partly drawn from the SpamAssassin "
        "corpus, so without this step SpamAssassin would not be an external test.",
        "",
        "## Results",
        "",
        "| Model | Test set | Precision | Recall | F1 | Accuracy | FPR | TP/FP/TN/FN |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline.md").write_text("\n".join(lines))
    (RESULTS / "baseline.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS / 'baseline.md'} in {time.time() - start:.0f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
