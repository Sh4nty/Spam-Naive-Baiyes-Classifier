"""Cross-check the vectorised models against a slow reference written
straight from the textbook, and test feature selection [IIR section 13.5]."""

import math
from collections import Counter

import numpy as np
import pytest

from spam_nb.features import chi_square, mutual_information, select_top_k
from spam_nb.naive_bayes import BernoulliNB, MultinomialNB
from spam_nb.tokenizer import tokenize

# ----------------------------------------------------------------------------
# Slow reference implementations (IIR Figures 13.2 and 13.3)
# ----------------------------------------------------------------------------


def reference_multinomial(train, labels, test, vocab):
    priors, cond = {}, {}
    for c in (0, 1):
        docs = [tokenize(d) for d, y in zip(train, labels) if y == c]
        priors[c] = len(docs) / len(train)
        counts = Counter(t for doc in docs for t in doc if t in vocab)
        total = sum(counts.values())
        cond[c] = {t: (counts[t] + 1) / (total + len(vocab)) for t in vocab}
    out = []
    for d in test:
        tokens = [t for t in tokenize(d) if t in vocab]
        out.append([math.log(priors[c]) + sum(math.log(cond[c][t]) for t in tokens)
                    for c in (0, 1)])
    return np.array(out)


def reference_bernoulli(train, labels, test, vocab):
    priors, cond = {}, {}
    for c in (0, 1):
        docs = [set(tokenize(d)) for d, y in zip(train, labels) if y == c]
        priors[c] = len(docs) / len(train)
        cond[c] = {t: (sum(t in doc for doc in docs) + 1) / (len(docs) + 2) for t in vocab}
    out = []
    for d in test:
        present = set(tokenize(d))
        out.append([
            math.log(priors[c]) + sum(
                math.log(cond[c][t]) if t in present else math.log(1 - cond[c][t])
                for t in vocab
            )
            for c in (0, 1)
        ])
    return np.array(out)


REFERENCE = {MultinomialNB: reference_multinomial, BernoulliNB: reference_bernoulli}


def random_corpus(seed, n_docs=60, vocab_size=40):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(vocab_size)]
    docs, labels = [], []
    for _ in range(n_docs):
        label = int(rng.random() < 0.5)
        # Spam prefers low ids, ham prefers high ids, with overlap.
        p = np.linspace(1, 3, vocab_size) if label == 0 else np.linspace(3, 1, vocab_size)
        doc = rng.choice(words, size=rng.integers(1, 30), p=p / p.sum())
        docs.append(" ".join(doc))
        labels.append(label)
    labels[0], labels[1] = 0, 1  # both classes present
    test = [" ".join(rng.choice(words + ["unseen"], size=rng.integers(0, 25)))
            for _ in range(30)]
    return docs, labels, test


@pytest.mark.parametrize("model_class", [MultinomialNB, BernoulliNB])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_implementation(model_class, seed):
    train, labels, test = random_corpus(seed)
    model = model_class().fit(train, labels)
    expected = REFERENCE[model_class](train, labels, test, set(model.vocabulary_))
    assert np.allclose(model.joint_log_likelihood(test), expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("model_class", [MultinomialNB, BernoulliNB])
@pytest.mark.parametrize("selector", ["frequency", "mi", "chi2"])
def test_selected_model_matches_reference_on_selected_vocabulary(model_class, selector):
    train, labels, test = random_corpus(3)
    model = model_class(selector=selector, k=10).fit(train, labels)
    assert len(model.vocabulary_) == 10
    expected = REFERENCE[model_class](train, labels, test, set(model.vocabulary_))
    assert np.allclose(model.joint_log_likelihood(test), expected, atol=1e-9, rtol=0)


# ----------------------------------------------------------------------------
# Textbook Examples 13.3 (MI) and 13.4 (chi-square): "export" in "poultry"
# ----------------------------------------------------------------------------

N11, N10, N01, N00 = 49, 27_652, 141, 774_106
# Row 1 = class of interest (poultry), row 0 = everything else.
DOC_FREQ = np.array([[N10], [N11]])
CLASS_COUNTS = (N10 + N00, N11 + N01)


def test_example_13_3_mutual_information():
    assert mutual_information(DOC_FREQ, CLASS_COUNTS)[0] == pytest.approx(0.0001105, abs=5e-8)


def test_example_13_4_chi_square():
    assert chi_square(DOC_FREQ, CLASS_COUNTS)[0] == pytest.approx(284, abs=1)


def test_degenerate_terms_score_zero():
    # A term in every document, or in no document, carries no information.
    doc_freq = np.array([[10, 0], [5, 0]])
    counts = (10, 5)
    assert mutual_information(doc_freq, counts) == pytest.approx([0, 0])
    assert chi_square(doc_freq, counts) == pytest.approx([0, 0])


# ----------------------------------------------------------------------------
# Selection behaviour
# ----------------------------------------------------------------------------

SEL_DOCS = [
    "free prize money common",
    "free offer cash common",
    "free prize win common",
    "meeting budget report common",
    "meeting project review common",
    "lunch meeting team common",
]
SEL_LABELS = [1, 1, 1, 0, 0, 0]


def test_select_top_k_is_stable_and_sorted():
    assert select_top_k(np.array([1.0, 3.0, 3.0, 0.5]), 2).tolist() == [1, 2]
    with pytest.raises(ValueError):
        select_top_k(np.array([1.0]), 0)


def test_frequency_keeps_most_common_terms():
    model = MultinomialNB(selector="frequency", k=3).fit(SEL_DOCS, SEL_LABELS)
    assert set(model.vocabulary_) == {"common", "free", "meeting"}


@pytest.mark.parametrize("selector", ["mi", "chi2"])
def test_mi_and_chi2_prefer_discriminative_terms(selector):
    model = MultinomialNB(selector=selector, k=2).fit(SEL_DOCS, SEL_LABELS)
    assert set(model.vocabulary_) == {"free", "meeting"}
    assert "common" not in model.vocabulary_


def test_k_larger_than_vocabulary_keeps_everything():
    full = MultinomialNB().fit(SEL_DOCS, SEL_LABELS)
    big = MultinomialNB(selector="chi2", k=10_000).fit(SEL_DOCS, SEL_LABELS)
    assert big.vocabulary_ == full.vocabulary_
    assert np.allclose(big.weights_, full.weights_)


def test_selector_and_k_must_come_together():
    with pytest.raises(ValueError):
        MultinomialNB(selector="mi")
    with pytest.raises(ValueError):
        MultinomialNB(k=5)
    with pytest.raises(ValueError, match="Unknown selector"):
        MultinomialNB(selector="tfidf", k=5).fit(SEL_DOCS, SEL_LABELS)
