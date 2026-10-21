"""Evaluate A7 rewrites against both victims.

    python scripts/evaluate_a7.py                       # every generations file
    python scripts/evaluate_a7.py data/a7/generations_X.jsonl

Test set = Enron test ham + the A7 sample's spam (defanged originals).
Each variant replaces the spam with its rewrites; a refused rewrite keeps
the original (a failed attack). Reports before/after P/R/F1 with 95% CIs,
ASR, edit cost, refusal rate, and which tokens explain the evasions.
Writes results/a7_<tag>.md and .json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spam_nb.experiment import MODELS, RESULTS, fit_victim, load_config, prepare_data  # noqa: E402
from spam_nb.llm_camouflage import evasion_attribution, token_change_rate  # noqa: E402
from spam_nb.metrics import evaluate  # noqa: E402
from spam_nb.tokenizer import tokenize  # noqa: E402

SAMPLE = ROOT / "data" / "a7" / "a7_sample.jsonl"
METRICS = ["precision", "recall", "f1"]


def ci(res, key, metric):
    lo, hi = res[key]["ci"][metric]
    return f"{res[key]['metrics'][metric]:.3f} [{lo:.3f}, {hi:.3f}]"


def evaluate_file(path: Path, data, models, cfg) -> tuple[str, dict]:
    sample = [json.loads(line) for line in open(SAMPLE)]
    originals = {row["id"]: row["original"] for row in sample}
    records = [json.loads(line) for line in open(path)]
    meta = records[0]
    variants = sorted({r["variant"] for r in records})

    ham = data["enron_test"][data["enron_test"].label == 0]["text"].tolist()
    spam = [row["original"] for row in sample]
    docs = ham + spam
    labels = [0] * len(ham) + [1] * len(spam)

    lines = [
        f"# A7 LLM camouflage: {meta['backend']} / {meta['model']}",
        "",
        f"Source: `{path.relative_to(ROOT)}` (prompt {meta.get('prompt_version', '?')}, "
        f"temperature {meta.get('temperature', '-')}). Test set: {len(ham):,} Enron test ham + "
        f"{len(spam)} sampled spam that both victims flag when clean (defanged, so the "
        "'before' column already includes any effect of defanging). Refused rewrites "
        "keep the original message.",
        "",
    ]
    summary = {"backend": meta["backend"], "model": meta["model"], "variants": {}}

    for variant in variants:
        recs = {r["id"]: r for r in records if r["variant"] == variant}
        missing = [i for i in originals if i not in recs]
        refused = sum(r["refused"] for r in recs.values())
        mapping = {
            originals[i]: (recs[i]["rewrite"] if i in recs and not recs[i]["refused"]
                           else originals[i])
            for i in originals
        }
        pairs = [(originals[i], mapping[originals[i]]) for i in originals]
        change = np.mean([token_change_rate(o, r) for o, r in pairs])
        growth = np.mean([len(tokenize(r)) / max(1, len(tokenize(o))) - 1 for o, r in pairs])
        kept = np.mean([recs[i]["placeholders_kept"] for i in recs]) if recs else 0.0

        lines += [
            f"## Variant: {variant}",
            "",
            f"- Rewrites: {len(recs)}/{len(originals)}"
            + (f" ({len(missing)} missing: not generated yet)" if missing else "")
            + f"; refused: {refused} ({refused / max(1, len(recs)):.1%})",
            f"- Placeholders kept (pitch still actionable): {kept:.1%}",
            f"- Mean normalised token edit distance: {change:.2f}; "
            f"mean length change: {growth:+.0%}",
            "",
            "| Model | Precision before | after | Recall before | after | "
            "F1 before | after | ASR |",
            "|---|---|---|---|---|---|---|---|",
        ]
        variant_summary = {"refused": refused, "n": len(recs), "token_change": change,
                           "placeholders_kept": kept, "models": {}}
        attributions = []
        for name, model in models.items():
            res = evaluate(model, docs, labels, attack=lambda msgs: [mapping[m] for m in msgs],
                           n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"])
            lines.append(
                f"| {name} | {ci(res, 'clean', 'precision')} | {ci(res, 'attacked', 'precision')} "
                f"| {ci(res, 'clean', 'recall')} | {ci(res, 'attacked', 'recall')} "
                f"| {ci(res, 'clean', 'f1')} | {ci(res, 'attacked', 'f1')} | {res['asr']:.3f} |"
            )
            variant_summary["models"][name] = {
                "clean": res["clean"]["metrics"], "attacked": res["attacked"]["metrics"],
                "clean_ci": res["clean"]["ci"], "attacked_ci": res["attacked"]["ci"],
                "asr": res["asr"],
            }
            evaded = [(o, r) for o, r in pairs if model.margin([r])[0] <= 0]
            if evaded:
                attr = evasion_attribution(model, *zip(*evaded), top=12)
                attributions.append((name, len(evaded), attr))

        lines.append("")
        for name, n_evaded, attr in attributions:
            removed = ", ".join(f"{t} ({v:.0f})" for t, v in attr["removed_spam_evidence"])
            added = ", ".join(f"{t} ({v:.0f})" for t, v in attr["added_ham_evidence"])
            lines += [
                f"**Why {n_evaded} messages evaded {name} NB** (summed change in "
                "log-odds contribution across evaded messages):",
                "",
                f"- Spam evidence the rewrite removed: {removed}",
                f"- Ham evidence the rewrite added: {added or '(none)'}",
                "",
            ]
        summary["variants"][variant] = variant_summary
    return "\n".join(lines), summary


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "data" / "a7").glob("generations_*.jsonl"))
    if not paths:
        print("No generations found. Run scripts/generate_a7.py first.")
        return 1
    cfg = load_config()
    data = prepare_data(cfg)
    models = {name: fit_victim(name, data, cfg) for name in MODELS}
    RESULTS.mkdir(exist_ok=True)
    for path in paths:
        report, summary = evaluate_file(path.resolve(), data, models, cfg)
        tag = path.stem.replace("generations_", "")
        (RESULTS / f"a7_{tag}.md").write_text(report)
        (RESULTS / f"a7_{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
        print(f"Wrote results/a7_{tag}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
