"""Exact replication of Manning, Raghavan & Schuetze, IIR Ch. 13.

Example 13.1 (multinomial NB) and Example 13.2 (Bernoulli NB) use the same
four training documents. Class c = "China" is encoded as label 1 and
class c-bar = "not China" as label 0, matching the project's convention
that the class of interest (spam) is the positive class.
"""

import numpy as np
import pytest

from spam_nb.naive_bayes import BernoulliNB, MultinomialNB

TRAIN_DOCS = [
    "Chinese Beijing Chinese",
    "Chinese Chinese Shanghai",
    "Chinese Macao",
    "Tokyo Japan Chinese",
]
TRAIN_LABELS = [1, 1, 1, 0]
TEST_DOC = "Chinese Chinese Chinese Tokyo Japan"

# The tokenizer lowercases, so vocabulary keys are lowercase.
HAM, SPAM = 0, 1  # column order of the model: [not China, China]


def _prob(model, cls, word):
    return float(np.exp(model.feature_log_prob_[cls, model.vocabulary_[word]]))


# --------------------------------------------------------------------------
# Example 13.1 - Multinomial NB
# --------------------------------------------------------------------------

@pytest.fixture
def multinomial():
    return MultinomialNB(alpha=1.0).fit(TRAIN_DOCS, TRAIN_LABELS)


def test_example_13_1_vocabulary_size(multinomial):
    # chinese, beijing, shanghai, macao, tokyo, japan
    assert len(multinomial.vocabulary_) == 6


def test_example_13_1_priors(multinomial):
    assert np.exp(multinomial.log_prior_[SPAM]) == pytest.approx(3 / 4)
    assert np.exp(multinomial.log_prior_[HAM]) == pytest.approx(1 / 4)


def test_example_13_1_conditional_probabilities(multinomial):
    # China class: 8 tokens, |V| = 6 -> denominator 14.
    assert _prob(multinomial, SPAM, "chinese") == pytest.approx(6 / 14)  # = 3/7
    assert _prob(multinomial, SPAM, "tokyo") == pytest.approx(1 / 14)
    assert _prob(multinomial, SPAM, "japan") == pytest.approx(1 / 14)
    # Not-China class: 3 tokens, |V| = 6 -> denominator 9.
    assert _prob(multinomial, HAM, "chinese") == pytest.approx(2 / 9)
    assert _prob(multinomial, HAM, "tokyo") == pytest.approx(2 / 9)
    assert _prob(multinomial, HAM, "japan") == pytest.approx(2 / 9)


def test_example_13_1_scores_and_decision(multinomial):
    scores = np.exp(multinomial.joint_log_likelihood([TEST_DOC])[0])

    expected_china = 3 / 4 * (3 / 7) ** 3 * (1 / 14) * (1 / 14)
    expected_not_china = 1 / 4 * (2 / 9) ** 3 * (2 / 9) * (2 / 9)

    assert scores[SPAM] == pytest.approx(expected_china, rel=1e-12)
    assert scores[HAM] == pytest.approx(expected_not_china, rel=1e-12)
    # Textbook values, as printed: ~0.0003 vs ~0.0001.
    assert round(scores[SPAM], 4) == 0.0003
    assert round(scores[HAM], 4) == 0.0001
    assert multinomial.predict([TEST_DOC]).tolist() == [SPAM]


# --------------------------------------------------------------------------
# Example 13.2 - Bernoulli NB
# --------------------------------------------------------------------------

@pytest.fixture
def bernoulli():
    return BernoulliNB(alpha=1.0).fit(TRAIN_DOCS, TRAIN_LABELS)


def test_example_13_2_conditional_probabilities(bernoulli):
    # P(t|c) = (N_ct + 1) / (N_c + 2), with N_china = 3 and N_not = 1.
    assert _prob(bernoulli, SPAM, "chinese") == pytest.approx(4 / 5)
    assert _prob(bernoulli, SPAM, "japan") == pytest.approx(1 / 5)
    assert _prob(bernoulli, SPAM, "tokyo") == pytest.approx(1 / 5)
    for word in ("beijing", "macao", "shanghai"):
        assert _prob(bernoulli, SPAM, word) == pytest.approx(2 / 5)

    assert _prob(bernoulli, HAM, "chinese") == pytest.approx(2 / 3)
    assert _prob(bernoulli, HAM, "japan") == pytest.approx(2 / 3)
    assert _prob(bernoulli, HAM, "tokyo") == pytest.approx(2 / 3)
    for word in ("beijing", "macao", "shanghai"):
        assert _prob(bernoulli, HAM, word) == pytest.approx(1 / 3)


def test_example_13_2_scores_and_decision(bernoulli):
    scores = np.exp(bernoulli.joint_log_likelihood([TEST_DOC])[0])

    # Present: chinese, japan, tokyo. Absent: beijing, macao, shanghai.
    expected_china = 3 / 4 * (4 / 5) * (1 / 5) * (1 / 5) * (1 - 2 / 5) ** 3
    expected_not_china = 1 / 4 * (2 / 3) * (2 / 3) * (2 / 3) * (1 - 1 / 3) ** 3

    assert scores[SPAM] == pytest.approx(expected_china, rel=1e-12)
    assert scores[HAM] == pytest.approx(expected_not_china, rel=1e-12)
    # Textbook values, as printed: ~0.005 vs ~0.022.
    assert round(scores[SPAM], 3) == 0.005
    assert round(scores[HAM], 3) == 0.022
    assert bernoulli.predict([TEST_DOC]).tolist() == [HAM]


def test_models_disagree_on_example_document(multinomial, bernoulli):
    """The point of the two examples: same data, different event models."""
    assert multinomial.predict([TEST_DOC])[0] != bernoulli.predict([TEST_DOC])[0]
