"""Training-time defences against poisoning and test-time score attacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from spam_nb.metrics import classification_metrics


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