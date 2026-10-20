"""Evaluation harness. Spam (label 1) is the positive class throughout.

P = TP / (TP + FP), R = TP / (TP + FN), F1 = 2PR / (P + R)  [IIR section 8.3]

Evasion attacks show up as a Recall drop, poisoning as a Precision drop, so
all three are always reported.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from spam_nb.tokenizer import tokenize

# An attack takes a list of spam messages and returns the modified messages,
# one for one. White-box attacks capture the model in a closure.
Attack = Callable[[list[str]], list[str]]

METRIC_NAMES = ("precision", "recall", "f1", "accuracy", "fpr")

# Max resample-x-document cells held in memory at once during bootstrapping.
BOOTSTRAP_CHUNK_CELLS = 20_000_000


def _as_binary(values, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=int)
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 (ham) and 1 (spam).")
    return arr


def _safe_div(num, den):
    """num / den, defined as 0 where den == 0 (e.g. no spam predicted)."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast(num, den).shape, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


def confusion(y_true, y_pred) -> dict[str, int]:
    """Confusion counts with spam as the positive class."""
    t = _as_binary(y_true, "y_true").astype(bool)
    p = _as_binary(y_pred, "y_pred").astype(bool)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have equal lengths.")
    return {
        "tp": int(np.sum(t & p)),
        "fp": int(np.sum(~t & p)),
        "tn": int(np.sum(~t & ~p)),
        "fn": int(np.sum(t & ~p)),
    }


def _metrics_from_counts(tp, fp, tn, fn) -> dict[str, np.ndarray]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    fpr = _safe_div(fp, fp + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
    }


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Precision, recall, F1, accuracy and false-positive rate.

    A ratio whose denominator is zero is reported as 0.0. In particular,
    if nothing is predicted as spam, precision, recall and F1 are all 0.
    """
    c = confusion(y_true, y_pred)
    m = _metrics_from_counts(c["tp"], c["fp"], c["tn"], c["fn"])
    return {name: float(value) for name, value in m.items()}


def bootstrap_ci(
    y_true,
    y_pred,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for every metric in METRIC_NAMES.

    The test set is resampled with replacement n_resamples times; all
    resamples are scored in one vectorised pass.
    """
    t = _as_binary(y_true, "y_true").astype(bool)
    p = _as_binary(y_pred, "y_pred").astype(bool)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have equal lengths.")
    if len(t) == 0:
        raise ValueError("Cannot bootstrap an empty test set.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = len(t)
    tp = np.empty(n_resamples)
    fp = np.empty(n_resamples)
    tn = np.empty(n_resamples)
    fn = np.empty(n_resamples)

    # Chunk the resamples to keep memory flat on large test sets.
    chunk = max(1, min(n_resamples, BOOTSTRAP_CHUNK_CELLS // max(n, 1)))
    for start in range(0, n_resamples, chunk):
        stop = min(start + chunk, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        bt, bp = t[idx], p[idx]
        tp[start:stop] = np.sum(bt & bp, axis=1)
        fp[start:stop] = np.sum(~bt & bp, axis=1)
        tn[start:stop] = np.sum(~bt & ~bp, axis=1)
        fn[start:stop] = np.sum(bt & ~bp, axis=1)

    samples = _metrics_from_counts(tp, fp, tn, fn)
    lo_q = 100 * (1 - confidence) / 2
    hi_q = 100 - lo_q
    return {
        name: (
            float(np.percentile(values, lo_q)),
            float(np.percentile(values, hi_q)),
        )
        for name, values in samples.items()
    }


def _summary(y_true, y_pred, n_resamples, seed) -> dict:
    result = {
        "metrics": classification_metrics(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
    }
    if n_resamples:
        result["ci"] = bootstrap_ci(y_true, y_pred, n_resamples, seed=seed)
    return result


def evaluate(
    model,
    documents: Sequence[str],
    labels: Sequence[int],
    attack: Attack | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Score a fitted model, optionally under an evasion attack.

    The attack is applied to spam messages only; ham is left untouched.

    Returns a dict with:
      clean            metrics, confusion counts and bootstrap CIs
      clean_margin     S(d) for every test document
    and, when an attack is given:
      attacked         metrics / confusion / CIs after the attack
      attacked_margin  S(d) after the attack
      delta            attacked minus clean, for every metric
      asr              attack success rate: share of spam the clean model
                       caught that the attack now lets through
      n_attackable     size of that set (spam correctly flagged when clean)
      tokens_added     mean extra tokens per attacked spam message
      length_increase  mean relative token-length increase per message
      attacked_documents  the modified test set, for inspection
    Set n_resamples=0 to skip bootstrapping (fast, for grid sweeps).
    """
    docs = [str(d) for d in documents]
    y = _as_binary(labels, "labels")
    if len(docs) != len(y):
        raise ValueError("Documents and labels must have equal lengths.")

    clean_margin = model.margin(docs)
    clean_pred = (clean_margin > 0).astype(int)
    result = {
        "clean": _summary(y, clean_pred, n_resamples, seed),
        "clean_margin": clean_margin,
    }
    if attack is None:
        return result

    spam_idx = np.flatnonzero(y == 1)
    original = [docs[i] for i in spam_idx]
    modified = list(attack(list(original)))
    if len(modified) != len(original):
        raise ValueError(
            f"Attack returned {len(modified)} messages for "
            f"{len(original)} inputs; it must map one to one."
        )

    attacked_docs = list(docs)
    for i, text in zip(spam_idx, modified):
        attacked_docs[i] = str(text)

    attacked_margin = clean_margin.copy()
    if len(spam_idx):
        attacked_margin[spam_idx] = model.margin([attacked_docs[i] for i in spam_idx])
    attacked_pred = (attacked_margin > 0).astype(int)

    attacked = _summary(y, attacked_pred, n_resamples, seed)
    attackable = (y == 1) & (clean_pred == 1)
    evaded = attackable & (attacked_pred == 0)

    before = np.array([len(tokenize(d)) for d in original], dtype=float)
    after = np.array([len(tokenize(d)) for d in modified], dtype=float)
    added = after - before

    result.update(
        {
            "attacked": attacked,
            "attacked_margin": attacked_margin,
            "delta": {
                name: attacked["metrics"][name] - result["clean"]["metrics"][name]
                for name in METRIC_NAMES
            },
            "asr": float(_safe_div(evaded.sum(), attackable.sum())),
            "n_attackable": int(attackable.sum()),
            "tokens_added": float(added.mean()) if len(added) else 0.0,
            "length_increase": float(_safe_div(added, np.maximum(before, 1)).mean())
            if len(added)
            else 0.0,
            "attacked_documents": attacked_docs,
        }
    )
    return result
