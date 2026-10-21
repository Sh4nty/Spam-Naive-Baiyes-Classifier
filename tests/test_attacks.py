import numpy as np
import pytest

from spam_nb.attacks import (
    breaking_point,
    common_word_padding,
    estimate_good_words,
    min_good_words,
    passive_good_word,
    random_padding,
    robustness_auc,
    whitebox_distinct,
    whitebox_repeat,
    word_frequencies,
)
from spam_nb.metrics import evaluate
from spam_nb.naive_bayes import BernoulliNB, MultinomialNB
from spam_nb.tokenizer import tokenize

HAM = [
    "meeting schedule project budget review",
    "project meeting tomorrow at the office",
    "please review the attached budget report",
    "lunch with the team tomorrow",
    "quarterly budget meeting notes attached",
    "office party friday team lunch",
]
SPAM = [
    "free prize claim now click here",
    "claim your free prize money now",
    "cheap pills discount offer click",
    "winner winner free money offer",
    "free cash offer click now",
    "discount pills cheap money",
]
TRAIN = HAM + SPAM
LABELS = [0] * len(HAM) + [1] * len(SPAM)
TEST_SPAM = [
    "free prize money click now free",
    "cheap pills offer free",
    "claim free cash winner",
    "free free free money money offer click",
]


@pytest.fixture(params=[MultinomialNB, BernoulliNB], ids=lambda m: m.__name__)
def model(request):
    return request.param().fit(TRAIN, LABELS)


def multinomial():
    return MultinomialNB().fit(TRAIN, LABELS)


# ----------------------------------------------------------------------------
# Common properties
# ----------------------------------------------------------------------------


def all_attacks(model, budget):
    words, _ = word_frequencies(HAM)
    good = [w for w, _ in estimate_good_words(HAM, SPAM, min_ham_docs=1)]
    attacks = {
        "A1-random": random_padding(words, budget, seed=1),
        "A1-common": common_word_padding(words, budget),
        "A2": passive_good_word(good, budget),
        "A3-distinct": whitebox_distinct(model, budget),
    }
    if isinstance(model, MultinomialNB):
        attacks["A3-repeat"] = whitebox_repeat(model, budget)
    return attacks


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_attacks_keep_payload_and_respect_budget(model, budget):
    for name, attack in all_attacks(model, budget).items():
        out = attack(list(TEST_SPAM))
        assert len(out) == len(TEST_SPAM), name
        for original, attacked in zip(TEST_SPAM, out):
            assert attacked.startswith(original), name  # payload intact
            added = len(tokenize(attacked)) - len(tokenize(original))
            assert added <= budget, name
            if name in ("A1-random", "A3-repeat"):
                assert added == budget, name  # these never run out of words
            if budget == 0:
                assert attacked == original, name


def test_random_padding_is_seeded():
    words, _ = word_frequencies(HAM)
    a = random_padding(words, 5, seed=3)(TEST_SPAM)
    assert a == random_padding(words, 5, seed=3)(TEST_SPAM)
    assert a != random_padding(words, 5, seed=4)(TEST_SPAM)


def test_word_frequencies_sorted_and_alphabetic():
    words, counts = word_frequencies(["b a a 7 c", "a b"])
    assert words == ["a", "b", "c"]
    assert counts.tolist() == [3, 2, 1]


# ----------------------------------------------------------------------------
# A2: passive estimate uses only the attacker's corpora
# ----------------------------------------------------------------------------


def test_estimated_good_words_are_hammy():
    ranked = estimate_good_words(HAM, SPAM, min_ham_docs=2)
    words = [w for w, _ in ranked]
    assert words[0] in {"meeting", "budget", "project", "team", "lunch", "office",
                        "attached", "review", "tomorrow"}
    assert "free" not in words  # never in >= 2 ham documents
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------------------
# A3 and the analytic bound
# ----------------------------------------------------------------------------


def test_whitebox_uses_most_negative_weights():
    m = multinomial()
    out = whitebox_distinct(m, 2)(["free"])[0]
    added = tokenize(out)[1:]
    hammiest = [str(m.feature_names_[i]) for i in np.argsort(m.weights_)[:2]]
    assert added == hammiest


def test_bernoulli_whitebox_skips_words_already_present():
    m = BernoulliNB().fit(TRAIN, LABELS)
    top = str(m.feature_names_[np.argsort(m.weights_)[0]])
    out = whitebox_distinct(m, 1)([f"free prize {top}"])[0]
    assert tokenize(out)[-1] != top


def test_bound_is_exact_for_distinct_attack(model):
    """A3 with b*(d) words flips d; with b*(d) - 1 it does not."""
    bstar = min_good_words(model, TEST_SPAM, mode="distinct")
    margins = model.margin(TEST_SPAM)
    for message, b, s in zip(TEST_SPAM, bstar, margins):
        if s <= 0:
            assert b == 0
            continue
        assert np.isfinite(b) and b >= 1
        b = int(b)
        assert model.margin(whitebox_distinct(model, b)([message]))[0] <= 0
        assert model.margin(whitebox_distinct(model, b - 1)([message]))[0] > 0


def test_bound_is_exact_for_repetition_attack():
    m = multinomial()
    bstar = min_good_words(m, TEST_SPAM, mode="repeat")
    for message, b in zip(TEST_SPAM, bstar):
        b = int(b)
        assert m.margin(whitebox_repeat(m, b)([message]))[0] <= 0
        assert m.margin(whitebox_repeat(m, b - 1)([message]))[0] > 0


def test_repetition_is_useless_against_bernoulli():
    m = BernoulliNB().fit(TRAIN, LABELS)
    with pytest.raises(ValueError):
        min_good_words(m, TEST_SPAM, mode="repeat")
    once = m.margin(whitebox_repeat(m, 1)(TEST_SPAM))
    many = m.margin(whitebox_repeat(m, 50)(TEST_SPAM))
    assert np.allclose(once, many)


def test_unflippable_message_reports_infinity():
    m = multinomial()
    huge = " ".join(["free"] * 500)
    assert np.isinf(min_good_words(m, [huge], mode="distinct", max_words=3)[0])


# ----------------------------------------------------------------------------
# Curve summaries and harness integration
# ----------------------------------------------------------------------------


def test_breaking_point_and_auc():
    budgets = [0, 10, 20, 40]
    f1 = [0.9, 0.8, 0.4, 0.1]
    assert breaking_point(budgets, f1, 0.9) == 20
    assert breaking_point(budgets, [0.9, 0.9, 0.9, 0.9], 0.9) is None
    assert robustness_auc(budgets, [0.9] * 4, 0.9) == pytest.approx(1.0)
    assert 0 < robustness_auc(budgets, f1, 0.9) < 1


def test_whitebox_beats_control_in_harness():
    m = multinomial()
    docs = HAM[:2] + TEST_SPAM
    labels = [0, 0] + [1] * len(TEST_SPAM)
    budget = int(min_good_words(m, TEST_SPAM, mode="distinct").max())
    words, _ = word_frequencies(["zebra yak xylophone walrus vulture"])
    control = evaluate(m, docs, labels, attack=random_padding(words, budget, seed=0), n_resamples=0)
    white = evaluate(m, docs, labels, attack=whitebox_distinct(m, budget), n_resamples=0)
    assert white["asr"] == 1.0  # the analytic budget flips every message
    assert control["asr"] == 0.0  # out-of-vocabulary padding does nothing
