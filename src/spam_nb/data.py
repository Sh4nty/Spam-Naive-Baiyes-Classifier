
from pathlib import Path
from typing import Tuple

import pandas as pd


def load_enron_spam(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Load and validate the Enron-Spam CSV."""
    df = pd.read_csv(path)

    required = {text_col, label_col, timestamp_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[text_col, label_col, timestamp_col]].copy()
    df = df.rename(
        columns={
            text_col: "text",
            label_col: "label",
            timestamp_col: "timestamp",
        }
    )

    df["text"] = df["text"].fillna("").astype(str)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    # Labels are represented consistently as integers:
    # spam = 1, ham = 0.
    if df["label"].dtype == object:
        normalized = df["label"].astype(str).str.strip().str.lower()
        label_map = {
            "spam": 1,
            "ham": 0,
            "1": 1,
            "0": 0,
        }
        df["label"] = normalized.map(label_map)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamps are missing or invalid. "
            "Resolve them before performing a temporal split."
        )

    if df["label"].isna().any():
        raise ValueError(
            "Some labels are invalid. Expected spam/ham or 1/0."
        )

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 (ham) or 1 (spam).")

    df["label"] = df["label"].astype(int)

    if len(df) < 2:
        raise ValueError("At least two emails are required.")

    # Stable sorting makes the split deterministic for tied timestamps.
    df = df.sort_values("timestamp", kind="mergesort")
    return df.reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: earlier emails train, later emails test."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    if len(df) < 2:
        raise ValueError("At least two emails are required.")

    ordered = df.sort_values("timestamp", kind="mergesort")
    ordered = ordered.reset_index(drop=True)

    split_idx = int(len(ordered) * train_fraction)
    split_idx = max(1, min(split_idx, len(ordered) - 1))

    train = ordered.iloc[:split_idx].copy()
    test = ordered.iloc[split_idx:].copy()

    assert train["timestamp"].max() <= test["timestamp"].min()

    return train, test

