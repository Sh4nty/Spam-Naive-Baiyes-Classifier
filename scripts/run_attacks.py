"""Good-word attacks A1-A3 against both NB variants on the Enron test set.

    python scripts/run_attacks.py [--seeds 5]

Writes:
  results/attacks_a1_a3.csv     one row per model x attack x budget (x seed)
  results/attacks_a1_a3.md      summary tables for the report
  results/attacks_f1_vs_budget.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spam_nb.attacks import (  # noqa: E402
    breaking_point,
    common_word_padding,
    estimate_good_words,
    min_good_words,
    passive_good_word,
    random_padding,
    robustness_auc,
    whitebox_distinct,
    whitebox_repeat,
    word_frequencies,
)
from spam_nb.experiment import (  # noqa: E402
    MODELS,
    RESULTS,
    fit_victim,
    load_config,
    prepare_data,
)
from spam_nb.metrics import evaluate  # noqa: E402

BUDGETS = [0, 5, 10, 20, 50, 100, 200, 500]
HEADLINE_BUDGET = 50
ATTACK_LABELS = {
    "A1-random": "A1 random padding (T1, control)",
    "A1-common": "A1 common-word padding (T1, control)",
    "A2": "A2 passive good-word (T2)",
    "A3-distinct": "A3 white-box, distinct words (T4)",
    "A3-repeat": "A3 white-box, one word repeated (T4)",
}


def build_attacks(model_name, model, pools, budget, seed):
    attacks = {
        "A1-random": random_padding(pools["random"], budget, seed=seed),
        "A1-common": common_word_padding(pools["common"], budget),
        "A2": passive_good_word(pools["good"], budget),
        "A3-distinct": whitebox_distinct(model, budget),
    }
    if model_name == "multinomial":
        attacks["A3-repeat"] = whitebox_repeat(model, budget)
    return attacks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5,
                        help="seeds for the randomised attack A1-random")
    args = parser.parse_args()

    cfg = load_config()
    start = time.time()
    data = prepare_data(cfg)
    test = data["enron_test"]
    docs, labels = test["text"].tolist(), test["label"].to_numpy()

    # Attacker-side resources: only the attacker's own public corpora.
    common, counts = word_frequencies(data["attacker_ham"]["text"])
    pools = {
        "common": common,
        "random": [w for w, c in zip(common, counts) if c >= 2],
        "good": [w for w, _ in estimate_good_words(
            data["attacker_ham"]["text"], data["attacker_spam"]["text"])],
    }

    rows, summaries, bstar_rows, examples = [], [], [], []
    for model_name in MODELS:
        model = fit_victim(model_name, data, cfg)
        clean = evaluate(model, docs, labels, n_resamples=0)
        clean_m = clean["clean"]["metrics"]
        attackable = (labels == 1) & (clean["clean_margin"] > 0)
        spam_docs = [d for d, a in zip(docs, attackable) if a]

        # Analytic minimum effort on the attackable set.
        modes = ["distinct"] + (["repeat"] if model_name == "multinomial" else [])
        for mode in modes:
            b = min_good_words(model, spam_docs, mode=mode)
            finite = b[np.isfinite(b)]
            bstar_rows.append({
                "model": model_name, "mode": mode,
                "median": float(np.median(finite)),
                "p10": float(np.percentile(finite, 10)),
                "p90": float(np.percentile(finite, 90)),
                "flippable_within_50": float(np.mean(b <= 50)),
                "flippable_within_500": float(np.mean(b <= 500)),
            })

        # What the two knowledge levels pick as "good words".
        a3_words = [str(model.feature_names_[i]) for i in np.argsort(model.weights_)
                    if str(model.feature_names_[i]).isalpha()][:10]
        a2_weights = [
            (w, model.weights_[model.vocabulary_[w]]) if w in model.vocabulary_ else (w, None)
            for w in pools["good"][:10]
        ]
        examples.append((model_name, a3_words, a2_weights))

        for budget in BUDGETS:
            seeds = range(args.seeds)
            for seed in seeds:
                for name, attack in build_attacks(model_name, model, pools, budget, seed).items():
                    if name != "A1-random" and seed > 0:
                        continue  # deterministic attacks run once
                    res = evaluate(model, docs, labels, attack=attack, n_resamples=0)
                    m = res["attacked"]["metrics"]
                    rows.append({
                        "model": model_name, "attack": name, "budget": budget, "seed": seed,
                        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                        "asr": res["asr"], "tokens_added": res["tokens_added"],
                        "clean_f1": clean_m["f1"], "clean_recall": clean_m["recall"],
                    })
            print(f"{model_name:12s} budget {budget:4d} done ({time.time() - start:.0f}s)")

        # Bootstrap CIs at the headline budget.
        for name, attack in build_attacks(model_name, model, pools, HEADLINE_BUDGET, 0).items():
            res = evaluate(model, docs, labels, attack=attack,
                           n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"])
            summaries.append({
                "model": model_name, "attack": name,
                "recall": res["attacked"]["metrics"]["recall"],
                "recall_ci": res["attacked"]["ci"]["recall"],
                "f1": res["attacked"]["metrics"]["f1"],
                "f1_ci": res["attacked"]["ci"]["f1"],
                "asr": res["asr"],
            })

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "attacks_a1_a3.csv", index=False)

    # Average the randomised attack over seeds for the curves.
    curves = df.groupby(["model", "attack", "budget"], as_index=False).agg(
        f1=("f1", "mean"), f1_sd=("f1", "std"), recall=("recall", "mean"),
        precision=("precision", "mean"), asr=("asr", "mean"), clean_f1=("clean_f1", "first"),
    )

    write_report(curves, summaries, bstar_rows, examples, args.seeds)
    plot(curves)
    print(f"Done in {time.time() - start:.0f}s")
    return 0


def write_report(curves, summaries, bstar_rows, examples, n_seeds):
    lines = [
        "# Good-word attacks A1-A3 (Enron test set)",
        "",
        "Victims trained on the Enron training split, full vocabulary, alpha = 1. "
        "Attacks append words to every spam message in the Enron test set; ham "
        "is untouched. Budget = tokens appended per message. ASR is measured on "
        "the spam each model flags when clean. "
        f"A1-random is averaged over {n_seeds} seeds; the other attacks are deterministic. "
        "A1 and A2 draw only on the attacker's own corpora (SpamAssassin ham and "
        "spam, disjoint from the victim's training data); A3 reads the victim's weights.",
        "",
        "## F1 by budget",
        "",
    ]
    for model in curves.model.unique():
        sub = curves[curves.model == model]
        table = sub.pivot(index="attack", columns="budget", values="f1")
        clean = sub.clean_f1.iloc[0]
        lines += [f"### {model} (clean F1 = {clean:.3f})", "",
                  "| Attack | " + " | ".join(f"b={b}" for b in table.columns)
                  + " | Breaking point | Robustness AUC |",
                  "|---|" + "---|" * (len(table.columns) + 2)]
        for attack in [a for a in ATTACK_LABELS if a in table.index]:
            f1 = table.loc[attack].to_numpy()
            bp = breaking_point(list(table.columns), f1, clean)
            auc = robustness_auc(list(table.columns), f1, clean)
            lines.append(
                f"| {ATTACK_LABELS[attack]} | " + " | ".join(f"{v:.3f}" for v in f1)
                + f" | {bp if bp is not None else '> 500'} | {auc:.3f} |"
            )
        lines.append("")

    lines += [
        f"## Headline: budget {HEADLINE_BUDGET} with 95% bootstrap CIs",
        "",
        "| Model | Attack | Recall | F1 | ASR |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {ATTACK_LABELS[s['attack']]} | {s['recall']:.3f} "
            f"[{s['recall_ci'][0]:.3f}, {s['recall_ci'][1]:.3f}] | {s['f1']:.3f} "
            f"[{s['f1_ci'][0]:.3f}, {s['f1_ci'][1]:.3f}] | {s['asr']:.3f} |"
        )

    lines += [
        "",
        "## Analytic minimum effort b*(d) on the attackable spam",
        "",
        "b*(d) = fewest appended good words that flip message d, computed from the "
        "true weights (distinct: greedy over the most negative w_t; repeat: "
        "ceil(S(d)/|w_min|)). The tests check it is exact.",
        "",
        "| Model | Mode | Median | 10th pct | 90th pct | Flippable with <= 50 | <= 500 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in bstar_rows:
        lines.append(
            f"| {r['model']} | {r['mode']} | {r['median']:.0f} | {r['p10']:.0f} | "
            f"{r['p90']:.0f} | {r['flippable_within_50']:.1%} | {r['flippable_within_500']:.1%} |"
        )

    lines += ["", "## What each attacker picks as 'good words'", ""]
    for model, a3, a2 in examples:
        a2_txt = ", ".join(
            f"{w} ({wt:+.2f})" if wt is not None else f"{w} (not in vocab)" for w, wt in a2
        )
        lines += [
            f"- **{model}**, A3 white-box top 10: {', '.join(a3)}",
            f"- **{model}**, A2 passive top 10 with the victim's true weight "
            f"(negative = hammy): {a2_txt}",
        ]
    lines.append("")
    (RESULTS / "attacks_a1_a3.md").write_text("\n".join(lines))


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(curves.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.2), sharey=True)
    styles = {
        "A1-random": ("#9e9e9e", "--"), "A1-common": ("#616161", ":"),
        "A2": ("#1f77b4", "-"), "A3-distinct": ("#d62728", "-"), "A3-repeat": ("#ff7f0e", "-."),
    }
    for ax, model in zip(np.atleast_1d(axes), models):
        sub = curves[curves.model == model]
        for attack, (color, ls) in styles.items():
            s = sub[sub.attack == attack]
            if s.empty:
                continue
            x = s.budget.replace(0, 1)  # log axis: plot b=0 at 1
            ax.plot(x, s.f1, ls, color=color, marker="o", ms=3, label=ATTACK_LABELS[attack])
        clean = sub.clean_f1.iloc[0]
        ax.axhline(clean / 2, color="black", lw=0.6, alpha=0.5)
        ax.text(1.05, clean / 2 + 0.02, "half of clean F1", fontsize=7, alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(f"{model} NB (clean F1 {clean:.3f})")
        ax.set_xlabel("Words appended per spam message (b=0 shown at 1)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("F1 on Enron test")
    # Legend on the panel with the most curves (multinomial has A3-repeat).
    max(axes, key=lambda a: len(a.get_lines())).legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "attacks_f1_vs_budget.png", dpi=150)


if __name__ == "__main__":
    sys.exit(main())
