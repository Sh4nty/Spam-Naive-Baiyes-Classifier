"""A8 training-data poisoning helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from spam_nb.tokenizer import tokenize


def ham_vocabulary(documents: Iterable[str], min_count: int = 1) -> list[str]:
    """Return alphabetic ham tokens ordered by descending document frequency."""
    if min_count < 1:
        raise ValueError("min_count must be at least 1.")

    counts = Counter(
        token
        for document in documents
        for token in set(tokenize(document))
        if token.isalpha()
    )
    return [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ]


def make_poison_messages(
    ham_documents: Iterable[str],
    count: int,
    words_per_message: int = 20,
    seed: int = 0,
) -> list[str]:
    """Create spam-labelled messages filled with words from a ham corpus."""
    if count < 0:
        raise ValueError("count must be non-negative.")
    if words_per_message < 1:
        raise ValueError("words_per_message must be at least 1.")

    vocabulary = ham_vocabulary(ham_documents)
    if count and not vocabulary:
        raise ValueError("ham_documents must contain at least one alphabetic token.")

    rng = np.random.default_rng(seed)
    return [
        " ".join(rng.choice(vocabulary, size=words_per_message, replace=True).tolist())
        for _ in range(count)
    ]


def inject_poisoning(
    documents: Sequence[str],
    labels: Sequence[int],
    ham_documents: Iterable[str],
    rate: float,
    words_per_message: int = 20,
    seed: int = 0,
) -> tuple[list[str], list[int]]:
    """Append ham-vocabulary messages labelled spam at a fraction of training size."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")
    if len(documents) != len(labels):
        raise ValueError("Documents and labels must have equal lengths.")
    if not np.isin(np.asarray(labels, dtype=int), [0, 1]).all():
        raise ValueError("Labels must be 0 (ham) or 1 (spam).")

    count = int(np.ceil(rate * len(documents)))
    poison = make_poison_messages(ham_documents, count, words_per_message, seed)
    return list(documents) + poison, list(labels) + [1] * count