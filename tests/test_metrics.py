import numpy as np
import pytest

from spam_nb.metrics import (
    bootstrap_ci,
    classification_metrics,
    confusion,
    evaluate,
)
from spam_nb.naive_bayes import BernoulliNB, MultinomialNB

# ----------------------------------------------------------------------------
# Point metrics (spam = 1 is positive)
# ----------------------------------------------------------------------------

Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  # TP=3 FN=1 FP=1 TN=5


def test_confusion_counts():
    assert confusion(Y_TRUE, Y_PRED) == {"tp": 3, "fp": 1, "tn": 5, "fn": 1}


def test_metrics_by_hand():
    m = classification_metrics(Y_TRUE, Y_PRED)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)
    assert m["accuracy"] == pytest.approx(8 / 10)
    assert m["fpr"] == pytest.approx(1 / 6)


def test_spam_is_the_positive_class():
    # Swapping the roles of the classes must change precision and recall.
    m = classification_metrics([1, 0, 0, 0], [1, 1, 0, 0])
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1.0)


def test_nothing_predicted_as_spam_gives_zero_not_nan():
    m = classification_metrics([1, 1, 0], [0, 0, 0])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert not any(np.isnan(v) for v in m.values())


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        classification_metrics([1, 2], [1, 0])
    with pytest.raises(ValueError):
        classification_metrics([1, 0, 1], [1, 0])


# ----------------------------------------------------------------------------
# Bootstrap CIs
# ----------------------------------------------------------------------------

def _noisy_predictions(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    flip = rng.random(n) < 0.1
    return y, np.where(flip, 1 - y, y)


def test_bootstrap_is_deterministic_for_a_seed():
    y, p = _noisy_predictions()
    assert bootstrap_ci(y, p, seed=7) == bootstrap_ci(y, p, seed=7)
    assert bootstrap_ci(y, p, seed=7) != bootstrap_ci(y, p, seed=8)


def test_bootstrap_interval_brackets_the_point_estimate():
    y, p = _noisy_predictions()
    point = classification_metrics(y, p)
    ci = bootstrap_ci(y, p)
    for name, (lo, hi) in ci.items():
        assert lo <= point[name] <= hi
        assert hi - lo < 0.06  # 2,000 documents -> tight interval


def test_bootstrap_perfect_classifier():
    y = [1, 0] * 50
    ci = bootstrap_ci(y, y)
    assert ci["f1"] == (1.0, 1.0)
    assert ci["fpr"] == (0.0, 0.0)


def test_bootstrap_chunking_gives_same_answer(monkeypatch):
    # Memory chunking on big test sets must not change the result.
    import spam_nb.metrics as metrics

    y, p = _noisy_predictions(n=300)
    one_chunk = bootstrap_ci(y, p, n_resamples=200, seed=3)

    monkeypatch.setattr(metrics, "BOOTSTRAP_CHUNK_CELLS", 300 * 7)  # 7 per chunk
    assert metrics.bootstrap_ci(y, p, n_resamples=200, seed=3) == one_chunk


# ----------------------------------------------------------------------------
# evaluate(): clean and under attack
# ----------------------------------------------------------------------------

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
    "budget meeting tomorrow",
    "team lunch review",
    "free prize money now",
    "cheap offer click here",
]
TEST_LABELS = [0, 0, 1, 1]


@pytest.fixture(params=[MultinomialNB, BernoulliNB], ids=lambda m: m.__name__)
def model(request):
    return request.param().fit(TRAIN_DOCS, TRAIN_LABELS)


def test_clean_evaluation_matches_predict(model):
    result = evaluate(model, TEST_DOCS, TEST_LABELS, n_resamples=100)
    expected = classification_metrics(TEST_LABELS, model.predict(TEST_DOCS))
    assert result["clean"]["metrics"] == expected
    assert "attacked" not in result
    assert set(result["clean"]["ci"]) == set(expected)


def test_identity_attack_changes_nothing(model):
    result = evaluate(model, TEST_DOCS, TEST_LABELS, attack=lambda d: d, n_resamples=0)
    assert result["asr"] == 0.0
    assert result["tokens_added"] == 0.0
    assert all(v == 0.0 for v in result["delta"].values())


def test_attack_touches_only_spam(model):
    seen = []

    def spy(docs):
        seen.extend(docs)
        return [d + " meeting" for d in docs]

    result = evaluate(model, TEST_DOCS, TEST_LABELS, attack=spy, n_resamples=0)
    assert seen == ["free prize money now", "cheap offer click here"]
    assert result["attacked_documents"][:2] == TEST_DOCS[:2]
    assert result["tokens_added"] == 1.0


def test_hammy_padding_evades_and_lowers_recall():
    model = MultinomialNB().fit(TRAIN_DOCS, TRAIN_LABELS)
    pad = " meeting budget review project tomorrow" * 3

    result = evaluate(
        model, TEST_DOCS, TEST_LABELS, attack=lambda d: [x + pad for x in d], n_resamples=0
    )

    assert result["clean"]["metrics"]["recall"] == 1.0
    assert result["n_attackable"] == 2
    assert result["asr"] == 1.0
    assert result["attacked"]["metrics"]["recall"] == 0.0
    assert result["delta"]["recall"] == -1.0
    assert (result["attacked_margin"][2:] < result["clean_margin"][2:]).all()


def test_attack_must_return_one_message_per_input(model):
    with pytest.raises(ValueError, match="one to one"):
        evaluate(model, TEST_DOCS, TEST_LABELS, attack=lambda d: d[:1], n_resamples=0)
