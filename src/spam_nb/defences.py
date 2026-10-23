"""Training-time defences against poisoning and test-time score attacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from spam_nb.metrics import classification_metrics
from spam_nb.tokenizer import tokenize


@dataclass(frozen=True)
class RoniResult:
    """Accepted training set and audit information from a RONI pass."""

    documents: list[str]
    labels: list[int]
    accepted: list[str]
    rejected: list[str]
    f1_deltas: list[float]


def roni_filter(
    documents: Sequence[str],
    labels: Sequence[int],
    candidates: Sequence[str],
    fit_model: Callable[[Sequence[str], Sequence[int]], object],
    validation_documents: Sequence[str],
    validation_labels: Sequence[int],
    max_f1_drop: float = 0.0,
) -> RoniResult:
    """Reject candidates whose sequential addition drops held-out F1 too far."""
    if len(documents) != len(labels):
        raise ValueError("Documents and labels must have equal lengths.")
    if len(validation_documents) != len(validation_labels):
        raise ValueError("Validation documents and labels must have equal lengths.")
    if max_f1_drop < 0:
        raise ValueError("max_f1_drop must be non-negative.")

    accepted = list(documents)
    accepted_labels = list(labels)
    model = fit_model(accepted, accepted_labels)
    current_f1 = classification_metrics(
        validation_labels, model.predict(validation_documents)
    )["f1"]
    kept_candidates: list[str] = []
    rejected_candidates: list[str] = []
    f1_deltas: list[float] = []

    for candidate in candidates:
        trial_documents = accepted + [candidate]
        trial_labels = accepted_labels + [1]
        trial_model = fit_model(trial_documents, trial_labels)
        trial_f1 = classification_metrics(
            validation_labels, trial_model.predict(validation_documents)
        )["f1"]
        delta = float(trial_f1 - current_f1)
        f1_deltas.append(delta)
        if delta < -max_f1_drop:
            rejected_candidates.append(candidate)
            continue
        accepted = trial_documents
        accepted_labels = trial_labels
        current_f1 = trial_f1
        kept_candidates.append(candidate)

    return RoniResult(
        documents=accepted,
        labels=accepted_labels,
        accepted=kept_candidates,
        rejected=rejected_candidates,
        f1_deltas=f1_deltas,
    )


def cap_term_frequency(text: str, cap: int = 1) -> str:
    """Keep at most `cap` occurrences of each token, preserving token order."""
    if cap < 1:
        raise ValueError("cap must be at least 1.")
    seen: dict[str, int] = {}
    kept = []
    for token in tokenize(text):
        count = seen.get(token, 0)
        if count < cap:
            kept.append(token)
            seen[token] = count + 1
    return " ".join(kept)


def capped_margin(model, documents: Sequence[str], cap: int = 1) -> np.ndarray:
    """Score documents after applying D2 term-frequency capping."""
    return model.margin([cap_term_frequency(document, cap) for document in documents])


def clipped_margin(model, documents: Sequence[str], tau: float) -> np.ndarray:
    """Score documents with D3 per-term weights clipped to [-tau, tau]."""
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    clipped = np.clip(model.weights_, -tau, tau)
    margins = []
    for document in documents:
        features = model._features(document)
        margins.append(model.bias_ + sum(clipped[index] * value for index, value in features.items()))
    return np.asarray(margins, dtype=float)


def segment_margin(model, documents: Sequence[str], segment_size: int = 50) -> np.ndarray:
    """Return D4's maximum spam margin over fixed-size token segments."""
    if segment_size < 1:
        raise ValueError("segment_size must be at least 1.")
    margins = []
    for document in documents:
        tokens = tokenize(document)
        segments = [
            " ".join(tokens[start:start + segment_size])
            for start in range(0, len(tokens), segment_size)
        ] or [""]
        margins.append(float(np.max(model.margin(segments))))
    return np.asarray(margins, dtype=float)