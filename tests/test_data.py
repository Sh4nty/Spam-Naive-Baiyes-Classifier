
import pandas as pd
import pytest

from spam_nb.data import load_enron_spam, temporal_split


def sample_dataframe():
    return pd.DataFrame({
        "text": ["old ham", "old spam", "new ham", "new spam", "latest"],
        "label": [0, 1, 0, 1, 0],
        "timestamp": pd.to_datetime([
            "2001-01-01",
            "2001-01-02",
            "2001-01-03",
            "2001-01-04",
            "2001-01-05",
        ], utc=True),
    })


def test_temporal_split_preserves_chronology():
    train, test = temporal_split(sample_dataframe(), 0.6)

    assert len(train) == 3
    assert len(test) == 2
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_temporal_split_does_not_shuffle():
    train, test = temporal_split(sample_dataframe(), 0.6)

    assert train["text"].tolist() == ["old ham", "old spam", "new ham"]
    assert test["text"].tolist() == ["new spam", "latest"]


def test_invalid_train_fraction():
    with pytest.raises(ValueError):
        temporal_split(sample_dataframe(), 1.0)


def test_load_rejects_invalid_labels(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({
        "text": ["hello", "world"],
        "label": ["spam", "unknown"],
        "timestamp": ["2001-01-01", "2001-01-02"],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="labels"):
        load_enron_spam(path)

