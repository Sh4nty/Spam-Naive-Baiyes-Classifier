"""Multinomial and Bernoulli Naive Bayes [IIR sections 13.2-13.3], NumPy only.

Documents are encoded once into flat arrays of (document index, term id)
pairs; class counts come from np.bincount and scores from a weighted
bincount over the same pairs, so nothing loops over tokens in Python
except the regex tokenizer itself.

Both models share one scoring form. For class c:

    log P(c, d) = base_[c] + sum over pairs (d, t) of value * term_[c, t]

Multinomial: one pair per token occurrence, base_ = log prior,
             term_ = log P(t|c).
Bernoulli:   one pair per distinct present term, base_ = log prior +
             sum_t log(1 - P(t|c)), term_ = log P(t|c) - log(1 - P(t|c)).

Spam (label 1) is the positive class. margin(d) = S(d) =
log P(spam, d) - log P(ham, d) = bias_ + sum value * weights_[t].
"""

from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Iterable

import numpy as np
import pandas as pd

from spam_nb.features import score_terms, select_top_k
from spam_nb.tokenizer import tokenize


class BaseNaiveBayes:
    """Shared binary text-classification functionality."""

    def __init__(
        self,
        alpha: float = 1.0,
        selector: str | None = None,
        k: int | None = None,
    ):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if (selector is None) != (k is None):
            raise ValueError("Give both selector and k, or neither.")

        self.alpha = float(alpha)
        self.selector = selector
        self.k = k
        self.classes_ = np.array([0, 1])
        self.vocabulary_: dict[str, int] = {}
        self.log_prior_: np.ndarray | None = None
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_all(documents) -> tuple[list[str], np.ndarray]:
        """Flat token list and the document index of each token."""
        token_lists = [tokenize(doc) for doc in documents]
        lengths = np.fromiter((len(t) for t in token_lists), dtype=np.int64,
                              count=len(token_lists))
        flat = list(chain.from_iterable(token_lists))
        doc_idx = np.repeat(np.arange(len(token_lists), dtype=np.int64), lengths)
        return flat, doc_idx

    def _encode(self, documents) -> tuple[np.ndarray, np.ndarray, int]:
        """(document index, term id) for every in-vocabulary token."""
        docs = list(documents)
        flat, doc_idx = self._tokenize_all(docs)
        if flat:
            ids = self._index_.get_indexer(np.array(flat, dtype=object)).astype(np.int64)
        else:
            ids = np.empty(0, dtype=np.int64)
        keep = ids >= 0  # out-of-vocabulary tokens are dropped
        return doc_idx[keep], ids[keep], len(docs)

    @staticmethod
    def _unique_pairs(doc_idx, ids, n_terms):
        """Distinct (document, term) pairs."""
        if doc_idx.size == 0:
            return doc_idx, ids
        # Hash-based de-duplication; order is irrelevant for bincount and is
        # ~8x faster than sorting with np.unique on millions of pairs.
        keys = pd.unique(doc_idx * n_terms + ids)
        return keys // n_terms, keys % n_terms

    def _design(self, documents):
        """(document index, term id, value) triples; see module docstring."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, documents, labels):
        docs = list(documents)
        y = np.asarray(list(labels), dtype=int)

        if len(docs) == 0:
            raise ValueError("Training data cannot be empty.")

        if len(docs) != len(y):
            raise ValueError("Documents and labels must have equal lengths.")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must be 0 (ham) or 1 (spam).")

        class_counts = np.bincount(y, minlength=2)
        if np.any(class_counts == 0):
            raise ValueError("Training data must contain both classes.")

        flat, doc_idx = self._tokenize_all(docs)
        if not flat:
            raise ValueError("Training vocabulary is empty.")

        # Sorted vocabulary, as in the original implementation.
        codes, uniques = pd.factorize(np.array(flat, dtype=object), sort=True)
        codes = codes.astype(np.int64)
        vocab = np.asarray(uniques, dtype=object)

        doc_freq = self._doc_freq(doc_idx, codes, y, len(vocab))

        if self.selector is not None:
            self.term_scores_ = score_terms(self.selector, doc_freq, class_counts)
            keep = select_top_k(self.term_scores_, min(self.k, len(vocab)))
            remap = np.full(len(vocab), -1, dtype=np.int64)
            remap[keep] = np.arange(len(keep))
            codes = remap[codes]
            mask = codes >= 0
            codes, doc_idx = codes[mask], doc_idx[mask]
            vocab = vocab[keep]
            doc_freq = doc_freq[:, keep]

        self.n_features_seen_ = len(docs)
        self.vocabulary_ = {word: i for i, word in enumerate(vocab)}
        self._index_ = pd.Index(vocab, dtype=object)
        self.feature_names_ = vocab
        self.class_count_ = class_counts
        self.doc_freq_ = doc_freq
        self.log_prior_ = np.log(class_counts / len(y))

        self._estimate(doc_idx, codes, y, len(vocab))

        self.bias_ = float(self._base_[1] - self._base_[0])
        self.weights_ = self._term_[1] - self._term_[0]
        self.is_fitted_ = True
        return self

    @classmethod
    def _doc_freq(cls, doc_idx, codes, y, n_terms):
        """(2, V): number of ham / spam documents containing each term."""
        d, t = cls._unique_pairs(doc_idx, codes, n_terms)
        return np.bincount(y[d] * n_terms + t, minlength=2 * n_terms).reshape(
            2, n_terms
        ).astype(np.float64)

    def _estimate(self, doc_idx, codes, y, n_terms):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before prediction.")

    def joint_log_likelihood(self, documents) -> np.ndarray:
        """Unnormalised log P(c, d), shape (n_docs, 2), columns [ham, spam]."""
        self._check_fitted()
        doc_idx, ids, values, n = self._design(documents)
        scores = np.tile(self._base_, (n, 1))
        for c in (0, 1):
            scores[:, c] += np.bincount(
                doc_idx, weights=values * self._term_[c, ids], minlength=n
            )
        return scores

    # Backwards-compatible private name.
    _joint_log_likelihood = joint_log_likelihood

    def margin(self, documents) -> np.ndarray:
        """Spam margin S(d) for each document; positive means spam."""
        self._check_fitted()
        doc_idx, ids, values, n = self._design(documents)
        return self.bias_ + np.bincount(
            doc_idx, weights=values * self.weights_[ids], minlength=n
        )

    def _features(self, document: str) -> Counter:
        """{term id: x_t} for one document."""
        _, ids, values, _ = self._design([document])
        features: Counter = Counter()
        for idx, value in zip(ids.tolist(), values.tolist()):
            features[idx] += value
        return features

    def token_contributions(self, document: str) -> list[tuple[str, float, float]]:
        """Per-token share of S(d) as (token, x_t, x_t * w_t).

        Sorted from most spam-pushing to most ham-pushing. The bias is not
        included; margin(d) == bias_ + sum of the contributions.
        """
        self._check_fitted()
        rows = [
            (str(self.feature_names_[idx]), float(x), float(x * self.weights_[idx]))
            for idx, x in self._features(document).items()
        ]
        return sorted(rows, key=lambda row: row[2], reverse=True)

    def predict_log_proba(self, documents) -> np.ndarray:
        """Return normalized log probabilities, columns [ham, spam]."""
        scores = self.joint_log_likelihood(documents)

        # Numerically stable log-sum-exp normalization.
        max_score = scores.max(axis=1, keepdims=True)
        log_normalizer = max_score + np.log(
            np.exp(scores - max_score).sum(axis=1, keepdims=True)
        )

        return scores - log_normalizer

    def predict_proba(self, documents) -> np.ndarray:
        return np.exp(self.predict_log_proba(documents))

    def predict(self, documents) -> np.ndarray:
        return self.joint_log_likelihood(documents).argmax(axis=1)

    def score(self, documents, labels) -> float:
        y = np.asarray(list(labels), dtype=int)
        predictions = self.predict(documents)

        if len(y) != len(predictions):
            raise ValueError("Documents and labels must have equal lengths.")

        return float(np.mean(predictions == y))


class MultinomialNB(BaseNaiveBayes):
    """Multinomial Naive Bayes with add-alpha smoothing [IIR Fig. 13.2]."""

    def _estimate(self, doc_idx, codes, y, n_terms):
        # T_ct: occurrences of term t in class c.
        word_counts = np.bincount(
            y[doc_idx] * n_terms + codes, minlength=2 * n_terms
        ).reshape(2, n_terms).astype(np.float64)

        smoothed_counts = word_counts + self.alpha
        denominators = smoothed_counts.sum(axis=1, keepdims=True)

        self.feature_count_ = word_counts
        self.feature_log_prob_ = np.log(smoothed_counts / denominators)
        self._base_ = self.log_prior_.copy()
        self._term_ = self.feature_log_prob_

    def _design(self, documents):
        doc_idx, ids, n = self._encode(documents)
        return doc_idx, ids, np.ones(ids.size), n


class BernoulliNB(BaseNaiveBayes):
    """Bernoulli Naive Bayes with add-alpha smoothing [IIR Fig. 13.3]."""

    def _estimate(self, doc_idx, codes, y, n_terms):
        # N_ct: documents of class c containing term t.
        present_counts = self.doc_freq_

        # P(word present | class), smoothed.
        self.feature_prob_ = (present_counts + self.alpha) / (
            self.class_count_[:, None] + 2.0 * self.alpha
        )
        self.feature_log_prob_ = np.log(self.feature_prob_)
        self.feature_log_not_prob_ = np.log1p(-self.feature_prob_)

        # Every term starts "absent"; a present term swaps log(1-p) for log p.
        self._base_ = self.log_prior_ + self.feature_log_not_prob_.sum(axis=1)
        self._term_ = self.feature_log_prob_ - self.feature_log_not_prob_

    def _design(self, documents):
        doc_idx, ids, n = self._encode(documents)
        doc_idx, ids = self._unique_pairs(doc_idx, ids, len(self.vocabulary_))
        return doc_idx, ids, np.ones(ids.size), n
