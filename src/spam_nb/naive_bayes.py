
from collections import Counter
from typing import Iterable

import numpy as np

from spam_nb.tokenizer import tokenize


class BaseNaiveBayes:
    """Shared binary text-classification functionality."""

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        self.alpha = float(alpha)
        self.classes_ = np.array([0, 1])
        self.vocabulary_: dict[str, int] = {}
        self.log_prior_: np.ndarray | None = None
        self.is_fitted_ = False

    def _tokenize_documents(
        self, documents: Iterable[str]
    ) -> list[list[str]]:
        return [tokenize(doc) for doc in documents]

    def _fit_common(self, documents, labels):
        docs = list(documents)
        y = np.asarray(list(labels), dtype=int)

        if len(docs) == 0:
            raise ValueError("Training data cannot be empty.")

        if len(docs) != len(y):
            raise ValueError("Documents and labels must have equal lengths.")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must be 0 (ham) or 1 (spam).")

        self.classes_ = np.array([0, 1])
        self.n_features_seen_ = len(docs)

        tokenized = self._tokenize_documents(docs)
        vocab = sorted({token for doc in tokenized for token in doc})
        self.vocabulary_ = {word: i for i, word in enumerate(vocab)}

        class_counts = np.array([(y == c).sum() for c in self.classes_])

        if np.any(class_counts == 0):
            raise ValueError("Training data must contain both classes.")

        self.log_prior_ = np.log(class_counts / len(y))
        self.is_fitted_ = True

        return tokenized, y

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before prediction.")

    def predict_log_proba(self, documents) -> np.ndarray:
        """Return normalized log probabilities, columns [ham, spam]."""
        self._check_fitted()

        scores = self._joint_log_likelihood(documents)

        # Numerically stable log-sum-exp normalization.
        max_score = scores.max(axis=1, keepdims=True)
        log_normalizer = max_score + np.log(
            np.exp(scores - max_score).sum(axis=1, keepdims=True)
        )

        return scores - log_normalizer

    def predict_proba(self, documents) -> np.ndarray:
        return np.exp(self.predict_log_proba(documents))

    def predict(self, documents) -> np.ndarray:
        return self._joint_log_likelihood(documents).argmax(axis=1)

    def score(self, documents, labels) -> float:
        y = np.asarray(list(labels), dtype=int)
        predictions = self.predict(documents)

        if len(y) != len(predictions):
            raise ValueError("Documents and labels must have equal lengths.")

        return float(np.mean(predictions == y))


class MultinomialNB(BaseNaiveBayes):
    """Multinomial Naive Bayes with add-alpha smoothing."""

    def fit(self, documents, labels):
        tokenized, y = self._fit_common(documents, labels)
        vocab_size = len(self.vocabulary_)

        if vocab_size == 0:
            raise ValueError("Training vocabulary is empty.")

        # One count per occurrence of each word.
        word_counts = np.zeros(
            (2, vocab_size), dtype=np.float64
        )

        for tokens, label in zip(tokenized, y):
            counts = Counter(tokens)
            for word, count in counts.items():
                word_counts[label, self.vocabulary_[word]] += count

        smoothed_counts = word_counts + self.alpha
        denominators = (
            smoothed_counts.sum(axis=1, keepdims=True)
        )

        self.feature_log_prob_ = np.log(
            smoothed_counts / denominators
        )
        return self

    def _joint_log_likelihood(self, documents):
        self._check_fitted()
        scores = np.tile(self.log_prior_, (len(documents), 1))

        for row, tokens in enumerate(documents):
            counts = Counter(tokenize(tokens))
            for word, count in counts.items():
                idx = self.vocabulary_.get(word)
                if idx is not None:
                    scores[row] += count * self.feature_log_prob_[:, idx]

        return scores


class BernoulliNB(BaseNaiveBayes):
    """Bernoulli Naive Bayes with add-alpha smoothing."""

    def fit(self, documents, labels):
        tokenized, y = self._fit_common(documents, labels)
        vocab_size = len(self.vocabulary_)

        if vocab_size == 0:
            raise ValueError("Training vocabulary is empty.")

        # Each word contributes at most once per document.
        present_counts = np.zeros(
            (2, vocab_size), dtype=np.float64
        )

        for tokens, label in zip(tokenized, y):
            for word in set(tokens):
                present_counts[label, self.vocabulary_[word]] += 1

        class_counts = np.array([(y == c).sum() for c in self.classes_])

        # P(word present | class), smoothed.
        self.feature_prob_ = (
            present_counts + self.alpha
        ) / (
            class_counts[:, None] + 2.0 * self.alpha
        )

        self.feature_log_prob_ = np.log(self.feature_prob_)
        self.feature_log_not_prob_ = np.log1p(-self.feature_prob_)

        return self

    def _joint_log_likelihood(self, documents):
        self._check_fitted()

        vocab_size = len(self.vocabulary_)
        scores = np.tile(self.log_prior_, (len(documents), 1))

        # Bernoulli NB includes the contribution of absent vocabulary terms.
        scores += self.feature_log_not_prob_.sum(axis=1)

        for row, document in enumerate(documents):
            present_indices = {
                self.vocabulary_[word]
                for word in tokenize(document)
                if word in self.vocabulary_
            }

            if present_indices:
                indices = list(present_indices)
                scores[row] += (
                    self.feature_log_prob_[:, indices]
                    - self.feature_log_not_prob_[:, indices]
                ).sum(axis=1)

        return scores

