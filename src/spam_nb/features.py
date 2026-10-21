"""Feature selection for binary NB [IIR section 13.5].

All three selectors are computed from the same 2x2 term-class contingency
table, built from document frequencies:

    N11 = spam documents containing t     N10 = ham documents containing t
    N01 = spam documents without t        N00 = ham documents without t

For a binary task MI and chi-square give the same ranking for either class,
so one ranking serves both. Frequency selection ranks by overall document
frequency.
"""

from __future__ import annotations

import numpy as np

SELECTORS = ("frequency", "mi", "chi2")


def _table(doc_freq: np.ndarray, class_counts: np.ndarray):
    """doc_freq: shape (2, V), docs per class containing each term
    (row 0 ham, row 1 spam). class_counts: (n_ham, n_spam)."""
    df = np.asarray(doc_freq, dtype=np.float64)
    n_ham, n_spam = (float(c) for c in class_counts)
    n11, n10 = df[1], df[0]
    n01, n00 = n_spam - n11, n_ham - n10
    return n11, n10, n01, n00


def frequency_scores(doc_freq, class_counts) -> np.ndarray:
    n11, n10, _, _ = _table(doc_freq, class_counts)
    return n11 + n10


def mutual_information(doc_freq, class_counts) -> np.ndarray:
    """Expected mutual information I(U; C), IIR equation 13.17 (bits)."""
    n11, n10, n01, n00 = _table(doc_freq, class_counts)
    n = n11 + n10 + n01 + n00
    n1_, n0_ = n11 + n10, n01 + n00  # docs with / without the term
    n_1, n_0 = n11 + n01, n10 + n00  # spam / ham docs

    def term(n_ec, n_e, n_c):
        with np.errstate(divide="ignore", invalid="ignore"):
            value = (n_ec / n) * np.log2(n * n_ec / (n_e * n_c))
        return np.where(n_ec > 0, value, 0.0)  # 0 log 0 = 0

    return (
        term(n11, n1_, n_1)
        + term(n01, n0_, n_1)
        + term(n10, n1_, n_0)
        + term(n00, n0_, n_0)
    )


def chi_square(doc_freq, class_counts) -> np.ndarray:
    """Chi-square statistic, IIR equation 13.19."""
    n11, n10, n01, n00 = _table(doc_freq, class_counts)
    n = n11 + n10 + n01 + n00
    num = n * (n11 * n00 - n10 * n01) ** 2
    den = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, ties broken by index (stable)."""
    if k <= 0:
        raise ValueError("k must be positive.")
    order = np.argsort(-np.asarray(scores), kind="stable")
    return np.sort(order[:k])


def score_terms(selector: str, doc_freq, class_counts) -> np.ndarray:
    if selector == "frequency":
        return frequency_scores(doc_freq, class_counts)
    if selector == "mi":
        return mutual_information(doc_freq, class_counts)
    if selector == "chi2":
        return chi_square(doc_freq, class_counts)
    raise ValueError(f"Unknown selector {selector!r}; choose from {SELECTORS}.")
