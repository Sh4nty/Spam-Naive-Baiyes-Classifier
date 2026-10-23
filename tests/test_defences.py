import pytest

from spam_nb.defences import roni_filter


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