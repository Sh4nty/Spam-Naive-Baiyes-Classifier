"""Run the remaining M5 integrations and write one compact report.

    python scripts/run_m5.py

The report uses the prepared Enron split and records A5/A6 with and without D1,
D2-D4 against A2/A3, A8 poisoning rates, and D5 RONI-lite decisions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spam_nb.attacks import estimate_good_words, whitebox_distinct  # noqa: E402
from spam_nb.defences import (  # noqa: E402
    capped_margin,
    clipped_margin,
    roni_filter,
    segment_margin,
)
from spam_nb.encoding import (  # noqa: E402
    hidden_html_padding,
    homoglyph_substitution,
    sanitize_text,
)
from spam_nb.experiment import (  # noqa: E402
    MODELS,
    RESULTS,
    fit_victim,
    load_config,
    prepare_data,
)
from spam_nb.metrics import classification_metrics, evaluate  # noqa: E402
from spam_nb.naive_bayes import BernoulliNB, MultinomialNB  # noqa: E402
from spam_nb.poisoning import make_poison_messages  # noqa: E402


def score(margins, labels) -> dict[str, float]:
    return classification_metrics(labels, (np.asarray(margins) > 0).astype(int))


def defended_metrics(model, documents, labels, margin_function) -> dict[str, float]:
    return score(margin_function(model, documents), labels)


def main() -> int:
    cfg = load_config()
    data = prepare_data(cfg)
    test = data["enron_test"]
    docs, labels = test.text.tolist(), test.label.to_numpy()
    train = data["enron_train"]
    rows: list[dict] = []
    poisoning_rows: list[dict] = []
    roni_rows: list[dict] = []

    for model_name, model_class in MODELS.items():
        model = fit_victim(model_name, data, cfg)
        clean = evaluate(model, docs, labels, n_resamples=0)["clean"]["metrics"]

        a5 = hidden_html_padding("meeting budget project office")
        a6_docs = [homoglyph_substitution(document) for document in docs]
        a5_docs = a5([document for document, label in zip(docs, labels) if label == 1])
        attacked_a5 = list(docs)
        attacked_a6 = list(docs)
        spam_indices = np.flatnonzero(labels == 1)
        for index, document in zip(spam_indices, a5_docs):
            attacked_a5[index] = document
        for index in spam_indices:
            attacked_a6[index] = a6_docs[index]

        for attack_name, attacked in (("A5", attacked_a5), ("A6", attacked_a6)):
            raw = score(model.margin(attacked), labels)
            d1 = score(model.margin([sanitize_text(document) for document in attacked]), labels)
            rows.extend([
                {"model": model_name, "experiment": attack_name, "defence": "none", **raw},
                {"model": model_name, "experiment": attack_name, "defence": "D1", **d1},
            ])

        good = [word for word, _ in estimate_good_words(
            data["attacker_ham"].text, data["attacker_spam"].text
        )]
        a2 = lambda messages: [message + "\n" + " ".join(good[:20]) for message in messages]
        a3 = whitebox_distinct(model, 20)
        for attack_name, attack in (("A2", a2), ("A3", a3)):
            attack_result = evaluate(model, docs, labels, attack=attack, n_resamples=0)
            attacked = attack_result["attacked_documents"]
            for defence, margin_function in (
                ("D2", lambda victim, values: capped_margin(victim, values, 1)),
                ("D3", lambda victim, values: clipped_margin(victim, values, 1.0)),
                ("D4", lambda victim, values: segment_margin(victim, values, 50)),
            ):
                rows.append({
                    "model": model_name,
                    "experiment": attack_name,
                    "defence": defence,
                    **defended_metrics(model, attacked, labels, margin_function),
                })

        for rate in (0.005, 0.01, 0.02, 0.05):
            count = int(np.ceil(rate * len(train)))
            poison = make_poison_messages(data["attacker_ham"].text, count, seed=cfg["seed"])
            poisoned_model = model_class(alpha=cfg["smoothing_alpha"]).fit(
                train.text.tolist() + poison, train.label.tolist() + [1] * count
            )
            poisoning_rows.append({
                "model": model_name,
                "rate": rate,
                "count": count,
                **score(poisoned_model.margin(docs), labels),
            })

        candidates = make_poison_messages(
            data["attacker_ham"].text, min(20, max(1, len(train) // 100)), seed=cfg["seed"]
        )
        result = roni_filter(
            train.text.tolist(), train.label.tolist(), candidates,
            lambda candidate_docs, candidate_labels: model_class(alpha=cfg["smoothing_alpha"]).fit(
                candidate_docs, candidate_labels
            ),
            test.text.tolist(), test.label.tolist(), max_f1_drop=0.01,
        )
        roni_rows.append({
            "model": model_name,
            "candidates": len(candidates),
            "accepted": len(result.accepted),
            "rejected": len(result.rejected),
            "max_f1_drop": 0.01,
        })

    RESULTS.mkdir(exist_ok=True)
    report = {
        "config": cfg,
        "scoring": rows,
        "poisoning": poisoning_rows,
        "roni": roni_rows,
    }
    (RESULTS / "m5_integrated.json").write_text(json.dumps(report, indent=2, default=float))

    frame = pd.DataFrame(rows)
    lines = [
        "# M5 integrated attack and defence results",
        "",
        "Enron temporal-per-class test split; spam is the positive class. "
        "Point estimates use the configured victim models and no bootstrap resampling.",
        "",
        "## A5/A6 and D1; A2/A3 and D2-D4",
        "",
        "| Model | Experiment | Defence | Precision | Recall | F1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['experiment']} | {row['defence']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines += ["", "## A8 poisoning", "", "| Model | Rate | Count | Precision | Recall | F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in poisoning_rows:
        lines.append(
            f"| {row['model']} | {row['rate']:.1%} | {row['count']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines += ["", "## D5 RONI-lite", "", "| Model | Candidates | Accepted | Rejected | Threshold |",
              "|---|---:|---:|---:|---:|"]
    for row in roni_rows:
        lines.append(
            f"| {row['model']} | {row['candidates']} | {row['accepted']} | "
            f"{row['rejected']} | {row['max_f1_drop']:.3f} |"
        )
    (RESULTS / "m5_integrated.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS / 'm5_integrated.md'}")
    print(f"Wrote {RESULTS / 'm5_integrated.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())