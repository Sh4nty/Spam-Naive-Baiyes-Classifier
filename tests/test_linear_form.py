"""The linear form S(d) = bias_ + sum_t x_t * weights_[t] must agree with
the probabilistic scorer, because every attack and defence relies on it."""

import numpy as np
import pytest

from spam_nb.naive_bayes import BernoulliNB, MultinomialNB

TRAIN_DOCS = [
    "meeting schedule project budget review",
    "project meeting tomorrow at the office",
    "please review the attached budget report",
    "lunch with the team tomorrow",
    "free prize claim now click here",
    "claim your free prize money now",
    "cheap pills discount offer click",
    "winner winner free money offer",
]
TRAIN_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]

TEST_DOCS = [
    "meeting about the budget",
    "free money click now",
    "project prize offer tomorrow",
    "completely unseen words only",
    "",
    "FREE FREE FREE prize prize",
]

MODELS = [MultinomialNB, BernoulliNB]


@pytest.fixture(params=MODELS, ids=lambda m: m.__name__)
def model(request):
    return request.param(alpha=1.0).fit(TRAIN_DOCS, TRAIN_LABELS)


def test_weights_shape_and_names(model):
    assert model.weights_.shape == (len(model.vocabulary_),)
    for word, idx in model.vocabulary_.items():
        assert model.feature_names_[idx] == word


def test_margin_equals_log_posterior_odds(model):
    log_proba = model.predict_log_proba(TEST_DOCS)
    expected = log_proba[:, 1] - log_proba[:, 0]
    assert np.allclose(model.margin(TEST_DOCS), expected, atol=1e-9)


def test_margin_equals_joint_log_likelihood_difference(model):
    jll = model.joint_log_likelihood(TEST_DOCS)
    assert np.allclose(model.margin(TEST_DOCS), jll[:, 1] - jll[:, 0], atol=1e-9)


def test_positive_margin_is_exactly_spam_prediction(model):
    assert ((model.margin(TEST_DOCS) > 0).astype(int) == model.predict(TEST_DOCS)).all()


def test_unseen_and_empty_documents_score_the_bias(model):
    margins = model.margin(["completely unseen words only", ""])
    assert margins == pytest.approx([model.bias_, model.bias_])


def test_contributions_sum_to_margin(model):
    for doc in TEST_DOCS:
        contributions = model.token_contributions(doc)
        total = model.bias_ + sum(c for _, _, c in contributions)
        assert total == pytest.approx(model.margin([doc])[0], abs=1e-9)


def test_contributions_sorted_spammiest_first(model):
    contributions = model.token_contributions("meeting free prize budget")
    values = [c for _, _, c in contributions]
    assert values == sorted(values, reverse=True)


def test_multinomial_one_extra_token_shifts_margin_by_its_weight():
    """Basis of the analytic breaking point b*(d) (proposal section 6.3)."""
    model = MultinomialNB().fit(TRAIN_DOCS, TRAIN_LABELS)
    doc = "free money click now"
    base = model.margin([doc])[0]

    for word in ("meeting", "budget", "prize"):
        w = model.weights_[model.vocabulary_[word]]
        assert model.margin([doc + " " + word])[0] == pytest.approx(base + w)
        # Repetition keeps paying: k copies shift the margin by k * w.
        assert model.margin([doc + (" " + word) * 5])[0] == pytest.approx(base + 5 * w)


def test_bernoulli_repetition_does_not_change_margin():
    model = BernoulliNB().fit(TRAIN_DOCS, TRAIN_LABELS)
    doc = "free money click now meeting"
    base = model.margin([doc])[0]
    assert model.margin([doc + " meeting meeting meeting"])[0] == pytest.approx(base)

    # A new distinct word shifts it by exactly delta_t, once.
    delta = model.weights_[model.vocabulary_["budget"]]
    assert model.margin([doc + " budget"])[0] == pytest.approx(base + delta)


def test_hammy_words_have_negative_weight(model):
    assert model.weights_[model.vocabulary_["meeting"]] < 0
    assert model.weights_[model.vocabulary_["free"]] > 0
