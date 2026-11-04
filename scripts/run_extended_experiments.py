"""Run A4, D6, S2 and the feature-selection robustness grid.

The runner uses only the prepared Enron/SpamAssassin data and writes
``results/extended_experiments.json`` and ``.md``.  Fresh-100 and Reuters
remain opt-in external corpora because the repository does not contain
licensed, labelled copies of those datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from spam_nb.attacks import active_good_word, whitebox_distinct
from spam_nb.defences import (
    attack_aware_retrain,
    certified_accuracy,
    certified_radius,
)
from spam_nb.experiment import RESULTS, fit_victim, load_config, prepare_data
from spam_nb.metrics import classification_metrics, evaluate
from spam_nb.naive_bayes import BernoulliNB, MultinomialNB


def main() -> int:
    cfg = load_config()
    data = prepare_data(cfg)
    train, test = data["enron_train"], data["enron_test"]
    docs, labels = test.text.tolist(), test.label.to_numpy()
    candidate_words = data["attacker_ham"].text.str.cat(sep=" ").split()[:100]
    output: dict[str, list[dict]] = {"a4": [], "d6": [], "s2": [], "feature_grid": []}

    for model_name, model_class in (("multinomial", MultinomialNB), ("bernoulli", BernoulliNB)):
        model = fit_victim(model_name, data, cfg)
        query_count = [0]

        def query(text):
            query_count[0] += 1
            return float(model.margin([text])[0])

        a4 = active_good_word(candidate_words, query, budget=20, max_queries=100)
        a4_result = evaluate(model, docs, labels, attack=a4, n_resamples=0)
        output["a4"].append({
            "model": model_name,
            "budget": 20,
            "max_queries": 100,
            "queries": query_count[0],
            "asr": a4_result["asr"],
            "f1": a4_result["attacked"]["metrics"]["f1"],
            "recall": a4_result["attacked"]["metrics"]["recall"],
        })

        def fit(candidate_docs, candidate_labels):
            return model_class(alpha=cfg["smoothing_alpha"]).fit(
                candidate_docs, candidate_labels
            )

        retrained = attack_aware_retrain(
            train.text.tolist(), train.label.tolist(), fit,
            whitebox_distinct(model, 5), rounds=2,
        )
        output["d6"].append({
            "model": model_name,
            "rounds": len(retrained.models),
            "training_rows_after": len(retrained.documents),
            "final_f1": classification_metrics(
                labels, retrained.models[-1].predict(docs)
            )["f1"],
        })

        for tau in (0.25, 0.5, 1.0, 2.0):
            output["s2"].append({
                "model": model_name,
                "tau": tau,
                "radius_median": float(np.median(certified_radius(model, docs, tau))),
                "accuracy_r0": certified_accuracy(model, docs, labels, tau, 0),
                "accuracy_r1": certified_accuracy(model, docs, labels, tau, 1),
                "accuracy_r5": certified_accuracy(model, docs, labels, tau, 5),
            })

        for selector in (None, "frequency", "mi", "chi2"):
            for k in (25, 100, 300, 1000):
                if selector is None and k != 25:
                    continue
                selected = model_class(
                    alpha=cfg["smoothing_alpha"],
                    selector=selector,
                    k=k if selector is not None else None,
                ).fit(train.text, train.label)
                clean = evaluate(selected, docs, labels, n_resamples=0)
                attacked = evaluate(
                    selected, docs, labels,
                    attack=whitebox_distinct(selected, 20),
                    n_resamples=0,
                )
                output["feature_grid"].append({
                    "model": model_name,
                    "selector": selector or "none",
                    "k": k if selector is not None else "all",
                    "clean_f1": clean["clean"]["metrics"]["f1"],
                    "attack_f1": attacked["attacked"]["metrics"]["f1"],
                    "asr": attacked["asr"],
                })

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "extended_experiments.json").write_text(
        json.dumps(output, indent=2)
    )
    lines = [
        "# Extended experiments",
        "",
        "All results use the prepared Enron temporal-per-class split. "
        "A4, D6, S2 and the selector grid are evaluated on the untouched test set.",
        "",
        "## A4 active black-box attack",
        "",
        "| Model | Queries | ASR | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in output["a4"]:
        lines.append(
            f"| {row['model']} | {row['queries']} | {row['asr']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines += ["", "## D6 attack-aware retraining", "", "| Model | Rounds | Rows after | F1 |",
              "|---|---:|---:|---:|"]
    for row in output["d6"]:
        lines.append(
            f"| {row['model']} | {row['rounds']} | {row['training_rows_after']} | "
            f"{row['final_f1']:.3f} |"
        )
    lines += ["", "## S2 certified radius", "", "| Model | Tau | Median radius | R@0 | R@1 | R@5 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in output["s2"]:
        lines.append(
            f"| {row['model']} | {row['tau']:.2f} | {row['radius_median']:.1f} | "
            f"{row['accuracy_r0']:.3f} | {row['accuracy_r1']:.3f} | {row['accuracy_r5']:.3f} |"
        )
    lines += ["", "## Feature-selection grid", "", "| Model | Selector | k | Clean F1 | Attack F1 | ASR |",
              "|---|---|---:|---:|---:|---:|"]
    for row in output["feature_grid"]:
        lines.append(
            f"| {row['model']} | {row['selector']} | {row['k']} | "
            f"{row['clean_f1']:.3f} | {row['attack_f1']:.3f} | {row['asr']:.3f} |"
        )
    (RESULTS / "extended_experiments.md").write_text("\n".join(lines) + "\n")
    print("Wrote extended experiment results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
