import numpy as np
import pytest

from spam_nb.defences import (
    attack_aware_retrain,
    certified_accuracy,
    certified_radius,
    roni_filter,
)


class ToyModel:
    def __init__(self, poison_present):
        self.poison_present = poison_present

    def predict(self, documents):
        if self.poison_present:
            return [1] * len(documents)
        return [1, 1, 0, 0]


def fit_toy_model(documents, labels):
    return ToyModel("poison" in documents)


def test_roni_rejects_f1_harming_candidate_and_keeps_benign_candidate():
    result = roni_filter(
        ["seed"],
        [0],
        ["benign", "poison"],
        fit_toy_model,
        ["a", "b", "c", "d"],
        [1, 1, 0, 0],
    )
    assert result.accepted == ["benign"]
    assert result.rejected == ["poison"]
    assert result.documents == ["seed", "benign"]
    assert result.labels == [0, 1]
    assert result.f1_deltas == [pytest.approx(0.0), pytest.approx(-1 / 3)]


def test_roni_validates_lengths_and_threshold():
    with pytest.raises(ValueError):
        roni_filter(["seed"], [], [], fit_toy_model, [], [])
    with pytest.raises(ValueError):
        roni_filter([], [], [], fit_toy_model, ["a"], [],)
    with pytest.raises(ValueError):
        roni_filter([], [], [], fit_toy_model, [], [], max_f1_drop=-0.1)


def test_attack_aware_retraining_adds_attacked_spam_each_round():
    result = attack_aware_retrain(
        ["ham", "spam"],
        [0, 1],
        lambda documents, labels: (list(documents), list(labels)),
        lambda messages: [message + " padding" for message in messages],
        rounds=2,
    )
    assert len(result.models) == 2
    assert result.documents == [
        "ham", "spam", "spam padding", "spam padding", "spam padding padding"
    ]
    assert result.labels == [0, 1, 1, 1, 1]


def test_attack_aware_retraining_can_rebuild_adaptive_attack():
    seen = []

    def factory(model):
        seen.append(model)
        return lambda messages: [message + f" round{len(seen)}" for message in messages]

    result = attack_aware_retrain(
        ["ham", "spam"], [0, 1],
        lambda documents, labels: (list(documents), list(labels)),
        lambda messages: list(messages),
        rounds=2,
        attack_factory=factory,
    )
    assert len(seen) == 2
    assert result.documents[-2:] == ["spam round2", "spam round1 round2"]


def test_certified_radius_and_accuracy():
    class Model:
        weights_ = np.array([-2.0, 1.0])
        bias_ = 3.0

        def _features(self, document):
            return {0: 1} if document == "spam" else {}

    model = Model()
    assert certified_radius(model, ["spam", "ham"], tau=1).tolist() == [2, 3]
    assert certified_accuracy(model, ["spam", "ham"], [1, 1], tau=1, radius=0) == 1.0

    with pytest.raises(ValueError):
        certified_radius(model, ["spam"], tau=0)