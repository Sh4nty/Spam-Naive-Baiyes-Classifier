import pytest

from spam_nb.poisoning import ham_vocabulary, inject_poisoning, make_poison_messages
from spam_nb.tokenizer import tokenize


HAM = [
    "meeting project budget",
    "project review meeting",
    "lunch budget report",
]


def test_ham_vocabulary_uses_document_frequency_and_stable_order():
    assert ham_vocabulary(HAM) == ["budget", "meeting", "project", "lunch", "report", "review"]
    assert ham_vocabulary(HAM, min_count=2) == ["budget", "meeting", "project"]


def test_poison_messages_are_reproducible_and_use_ham_vocabulary():
    first = make_poison_messages(HAM, count=3, words_per_message=4, seed=7)
    assert first == make_poison_messages(HAM, count=3, words_per_message=4, seed=7)
    assert all(set(tokenize(message)) <= set(ham_vocabulary(HAM)) for message in first)


def test_inject_poisoning_uses_ceil_rate_and_spam_labels():
    documents = ["ham"] * 10
    labels = [0] * 10
    poisoned, poisoned_labels = inject_poisoning(
        documents, labels, HAM, rate=0.05, words_per_message=2, seed=1
    )
    assert len(poisoned) == 11
    assert poisoned_labels == [0] * 10 + [1]


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_inject_poisoning_rejects_invalid_rate(rate):
    with pytest.raises(ValueError):
        inject_poisoning(["ham"], [0], HAM, rate)