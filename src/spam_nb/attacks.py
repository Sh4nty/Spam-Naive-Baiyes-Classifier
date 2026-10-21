"""Good-word evasion attacks A1-A3 and the analytic breaking point.

Every attack builder returns an Attack: a function from a list of spam
messages to the modified list (same length, same order), which plugs
straight into spam_nb.metrics.evaluate(). The spam payload is never
removed: words are only appended, so the original pitch stays intact.

Threat models (proposal section 6.2):
  A1  T1 passive oblivious  random / common-word padding (negative control)
  A2  T2 passive gray-box   good words estimated from the attacker's own
                            public ham and spam; no access to the victim
  A3  T4 white-box          good words chosen from the victim's true
                            weights; gives the oracle lower bound

Budget unit: tokens appended per message.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Sequence

import numpy as np

from spam_nb.tokenizer import tokenize

Attack = Callable[[list[str]], list[str]]


def _append(message: str, words: Sequence[str]) -> str:
    return message + ("\n" + " ".join(words) if len(words) else "")


# ----------------------------------------------------------------------------
# A1: random / common-word padding (control)
# ----------------------------------------------------------------------------


def word_frequencies(texts: Iterable[str], min_count: int = 1) -> tuple[list[str], np.ndarray]:
    """Tokens in a corpus by descending count (ties alphabetical)."""
    counts = Counter(tok for text in texts for tok in tokenize(text))
    items = sorted(
        ((w, c) for w, c in counts.items() if c >= min_count and w.isalpha()),
        key=lambda wc: (-wc[1], wc[0]),
    )
    return [w for w, _ in items], np.array([c for _, c in items], dtype=float)


def random_padding(pool: Sequence[str], budget: int, seed: int = 0) -> Attack:
    """A1a: append `budget` words drawn uniformly at random from `pool`."""
    pool = list(pool)

    def attack(messages: list[str]) -> list[str]:
        rng = np.random.default_rng(seed)
        return [
            _append(m, [pool[i] for i in rng.integers(0, len(pool), budget)])
            for m in messages
        ]

    return attack


def common_word_padding(ranked_words: Sequence[str], budget: int) -> Attack:
    """A1b: append the `budget` most frequent words of a generic corpus."""
    words = list(ranked_words[:budget])
    return lambda messages: [_append(m, words) for m in messages]


# ----------------------------------------------------------------------------
# A2: passive good-word attack (Lowd & Meek 2005)
# ----------------------------------------------------------------------------


def estimate_good_words(
    attacker_ham: Iterable[str],
    attacker_spam: Iterable[str],
    min_ham_docs: int = 5,
) -> list[tuple[str, float]]:
    """Rank words by the attacker's own estimate of how 'hammy' they are.

    score_t = log P(t|ham) - log P(t|spam), with add-one smoothing over the
    attacker's corpora, keeping only alphabetic words that occur in at least
    `min_ham_docs` ham documents (rare words are unreliable estimates).
    Highest score first.
    """
    ham_docs = [set(tokenize(t)) for t in attacker_ham]
    spam_docs = [set(tokenize(t)) for t in attacker_spam]
    ham_df = Counter(w for d in ham_docs for w in d)
    spam_df = Counter(w for d in spam_docs for w in d)
    n_ham, n_spam = len(ham_docs), len(spam_docs)

    scored = [
        (w, float(np.log((ham_df[w] + 1) / (n_ham + 2)) - np.log((spam_df[w] + 1) / (n_spam + 2))))
        for w, c in ham_df.items()
        if c >= min_ham_docs and w.isalpha()
    ]
    return sorted(scored, key=lambda ws: (-ws[1], ws[0]))


def passive_good_word(good_words: Sequence[str], budget: int) -> Attack:
    """A2: append the attacker's top-`budget` estimated good words."""
    words = list(good_words[:budget])
    return lambda messages: [_append(m, words) for m in messages]


# ----------------------------------------------------------------------------
# A3: white-box good-word attack
# ----------------------------------------------------------------------------


def _hammiest_ids(model) -> np.ndarray:
    """Vocabulary ids with negative weight, most negative first.
    Only alphabetic words: an attacker appends words, not numbers."""
    order = np.argsort(model.weights_, kind="stable")
    order = order[model.weights_[order] < 0]
    alpha = np.array([str(model.feature_names_[i]).isalpha() for i in order], dtype=bool)
    return order[alpha]


def _is_bernoulli(model) -> bool:
    return hasattr(model, "feature_log_not_prob_")


def whitebox_distinct(model, budget: int) -> Attack:
    """A3(i): append the `budget` distinct words with the most negative w_t.

    For Bernoulli NB a word already in the message contributes nothing new,
    so it is skipped and the next hammy word is used instead.
    """
    ranked = _hammiest_ids(model)
    names = model.feature_names_
    bernoulli = _is_bernoulli(model)

    def attack(messages: list[str]) -> list[str]:
        if budget <= 0:
            return list(messages)
        out = []
        for m in messages:
            if bernoulli:
                present = set(tokenize(m))
                words = []
                for i in ranked:
                    w = str(names[i])
                    if w not in present:
                        words.append(w)
                        if len(words) == budget:
                            break
            else:
                words = [str(names[i]) for i in ranked[:budget]]
            out.append(_append(m, words))
        return out

    return attack


def whitebox_repeat(model, budget: int) -> Attack:
    """A3(ii): repeat the single most negative-weight word `budget` times.

    Effective against multinomial NB only; Bernoulli ignores repetition.
    """
    word = str(model.feature_names_[_hammiest_ids(model)[0]])
    return lambda messages: [_append(m, [word] * budget) for m in messages]


# ----------------------------------------------------------------------------
# Analytic breaking point b*(d)
# ----------------------------------------------------------------------------


def min_good_words(
    model,
    messages: Sequence[str],
    mode: str = "distinct",
    max_words: int = 5000,
) -> np.ndarray:
    """Exact minimum number of appended good words that flips each message.

    A message is spam iff S(d) > 0, and appending word t changes S(d) by
    w_t (multinomial) or delta_t once (Bernoulli, if t is absent), so:

      distinct: b*(d) = min{ b : sum of the b largest |w_t| over usable
                hammy words >= S(d) }   (greedy is optimal: all terms add)
      repeat:   b*(d) = ceil(S(d) / |w_min|)   (multinomial only)

    Returns 0 for messages already classified ham, and np.inf when the
    message cannot be flipped with at most `max_words` distinct words.
    """
    if mode not in ("distinct", "repeat"):
        raise ValueError("mode must be 'distinct' or 'repeat'.")
    bernoulli = _is_bernoulli(model)
    if mode == "repeat" and bernoulli:
        raise ValueError("Repetition does not change a Bernoulli NB score.")

    margins = model.margin(list(messages))
    ranked = _hammiest_ids(model)[:max_words]
    gains = -model.weights_[ranked]  # positive, descending
    out = np.zeros(len(margins))

    if mode == "repeat":
        flagged = margins > 0
        out[flagged] = np.ceil(margins[flagged] / gains[0])
        return out

    cumulative = np.cumsum(gains)
    rank_of = {int(i): r for r, i in enumerate(ranked)}
    for row, (message, s) in enumerate(zip(messages, margins)):
        if s <= 0:
            continue
        if bernoulli:
            # Words already present add nothing; drop them from the ranking.
            present = [rank_of[i] for i in model._features(message) if i in rank_of]
            if present:
                keep = np.ones(len(gains), dtype=bool)
                keep[present] = False
                cum = np.cumsum(gains[keep])
            else:
                cum = cumulative
        else:
            cum = cumulative
        # Smallest b with cum[b-1] >= s, i.e. new margin s - cum <= 0 (ham).
        b = int(np.searchsorted(cum, s, side="left")) + 1
        out[row] = b if b <= len(cum) else np.inf
    return out


def breaking_point(budgets: Sequence[int], f1: Sequence[float], clean_f1: float):
    """Smallest budget at which F1 falls below half its clean value, or None."""
    for b, f in zip(budgets, f1):
        if f < clean_f1 / 2:
            return b
    return None


def robustness_auc(budgets: Sequence[int], f1: Sequence[float], clean_f1: float) -> float:
    """Area under the normalised F1-vs-budget curve, budget scaled to [0, 1]."""
    x = np.asarray(budgets, dtype=float)
    y = np.asarray(f1, dtype=float) / clean_f1 if clean_f1 else np.zeros(len(budgets))
    if x[-1] == x[0]:
        return float(y[0])
    return float(np.trapezoid(y, (x - x[0]) / (x[-1] - x[0])))
