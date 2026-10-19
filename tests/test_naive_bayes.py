
import numpy as np
import pytest

from spam_nb.naive_bayes import MultinomialNB, BernoulliNB


@pytest.fixture
def training_data():
    documents = [
        "meeting schedule project",
        "project meeting tomorrow",
        "free prize claim now",
        "claim your free prize",
    ]
    labels = [0, 0, 1, 1]
    return documents, labels


@pytest.mark.parametrize("model_class", [MultinomialNB, BernoulliNB])
def test_fit_and_predict(model_class, training_data):
    documents, labels = training_data
    model = model_class(alpha=1.0).fit(documents, labels)

    predictions = model.predict([
        "meeting project",
        "free prize claim",
    ])

    assert predictions.tolist() == [0, 1]


@pytest.mark.parametrize("model_class", [MultinomialNB, BernoulliNB])
def test_probabilities_are_valid(model_class, training_data):
    documents, labels = training_data
    model = model_class().fit(documents, labels)

    probabilities = model.predict_proba(["meeting", "free prize"])

    assert probabilities.shape == (2, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("model_class", [MultinomialNB, BernoulliNB])
def test_unseen_words_do_not_crash(model_class, training_data):
    documents, labels = training_data
    model = model_class().fit(documents, labels)

    predictions = model.predict(["completely unseen vocabulary"])

    assert predictions.shape == (1,)
    assert predictions[0] in [0, 1]


@pytest.mark.parametrize("model_class", [MultinomialNB, BernoulliNB])
def test_predict_before_fit_raises(model_class):
    model = model_class()

    with pytest.raises(RuntimeError):
        model.predict(["hello"])


@pytest.mark.parametrize("model_class", [MultinomialNB, BernoulliNB])
def test_both_classes_are_required(model_class):
    with pytest.raises(ValueError, match="both classes"):
        model_class().fit(["hello", "world"], [0, 0])


def test_multinomial_counts_repeated_words(training_data):
    documents, labels = training_data
    model = MultinomialNB().fit(documents, labels)

    meeting_idx = model.vocabulary_["meeting"]

    # Both ham emails contain "meeting", so the total is two.
    assert model.feature_log_prob_[0, meeting_idx] == pytest.approx(
        np.log((2 + 1) / (5 + len(model.vocabulary_)))
    )


def test_bernoulli_counts_presence_not_frequency():
    model = BernoulliNB().fit(
        ["offer offer offer", "ordinary meeting", "offer prize", "meeting"],
        [1, 0, 1, 0],
    )

    offer_idx = model.vocabulary_["offer"]

    # Both spam documents contain "offer"; repetitions count only once.
    assert model.feature_prob_[1, offer_idx] == pytest.approx(3 / 4)

