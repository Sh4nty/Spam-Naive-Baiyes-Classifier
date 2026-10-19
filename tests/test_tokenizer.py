
from spam_nb.tokenizer import tokenize


def test_tokenize_lowercases_text():
    assert tokenize("FREE Money NOW") == ["free", "money", "now"]


def test_tokenize_ignores_punctuation():
    assert tokenize("hello!!! world...") == ["hello", "world"]


def test_tokenize_handles_none():
    assert tokenize(None) == []


def test_tokenize_numbers():
    assert tokenize("Offer 123") == ["offer", "123"]

