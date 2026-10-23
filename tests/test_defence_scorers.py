import numpy as np
import pytest

from spam_nb.defences import cap_term_frequency, capped_margin, clipped_margin, segment_margin
from spam_nb.naive_bayes import MultinomialNB


TRAIN = [
    "meeting project budget",
    "project review meeting",
    "free prize claim",
    "cheap pills offer",
]
LABELS = [0, 0, 1, 1]


@pytest.fixture
def model():
    return MultinomialNB().fit(TRAIN, LABELS)


def test_cap_term_frequency_preserves_order_and_limits_repeats():
    assert cap_term_frequency("free budget free free meeting budget", 1) == "free budget meeting"
    assert cap_term_frequency("free free free", 2) == "free free"


def test_defence_scorers_match_their_definitions(model):
    document = "free free free budget budget meeting"
    assert capped_margin(model, [document], 1)[0] == pytest.approx(
        model.margin(["free budget meeting"])[0]
    )
    expected = model.bias_ + sum(
        np.clip(model.weights_[index], -0.1, 0.1) * value
        for index, value in model._features(document).items()
    )
    assert clipped_margin(model, [document], 0.1)[0] == pytest.approx(expected)

    segments = ["free free", "free budget", "budget meeting"]
    assert segment_margin(model, [document], 2)[0] == pytest.approx(
        model.margin(segments).max()
    )


@pytest.mark.parametrize("value", [0, -1])
def test_defence_scorers_reject_invalid_parameters(model, value):
    with pytest.raises(ValueError):
        cap_term_frequency("a", value)
    with pytest.raises(ValueError):
        capped_margin(model, ["a"], value)
    with pytest.raises(ValueError):
        segment_margin(model, ["a"], value)

    if value < 0:
        with pytest.raises(ValueError):
            clipped_margin(model, ["a"], value)