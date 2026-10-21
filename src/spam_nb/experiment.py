"""Shared experiment setup: config, prepared corpora, fitted victims.

    from spam_nb.experiment import load_config, prepare_data, fit_victim
    cfg = load_config()
    data = prepare_data(cfg)
    model = fit_victim("multinomial", data, cfg)

Prepared data (cached in data/processed/ after the first run):

    enron_train   Enron, oldest 70% of each class (victim training data)
    enron_test    Enron, newest 30% of each class, minus near-duplicates
                  of enron_train (in-domain test set; attack target)
    sa_test       SpamAssassin, minus near-duplicates of enron_train
                  (external test set)
    attacker_ham  SpamAssassin ham: the passive attacker's public ham pool,
                  disjoint from the victim's training data
    attacker_spam SpamAssassin spam (deduplicated): the attacker's own spam
    dedup         how many rows were removed and why
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from spam_nb.data import (
    drop_near_duplicates,
    load_enron_csv,
    load_spamassassin,
    temporal_split,
)
from spam_nb.naive_bayes import BernoulliNB, MultinomialNB

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

MODELS = {"multinomial": MultinomialNB, "bernoulli": BernoulliNB}


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config" / "seed.json") as fh:
        return json.load(fh)


def _cache_key(cfg: dict) -> str:
    parts = [
        cfg["split_strategy"],
        str(cfg["train_fraction"]),
        str(cfg.get("dedup_threshold", 0.8)),
    ]
    return "prepared_" + "_".join(p.replace(".", "p") for p in parts) + ".pkl"


def prepare_data(cfg: dict | None = None, use_cache: bool = True) -> dict:
    cfg = cfg or load_config()
    cache = PROCESSED / _cache_key(cfg)
    if use_cache and cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    enron_path = RAW / "enron_spam_data.zip"
    sa_path = RAW / "spamassassin"
    if not enron_path.exists() or not sa_path.exists():
        raise FileNotFoundError(
            "Raw data missing. Run: python scripts/fetch_data.py"
        )

    threshold = cfg.get("dedup_threshold", 0.8)
    enron = load_enron_csv(enron_path)
    stratify = cfg["split_strategy"] == "temporal_per_class"
    train, test = temporal_split(enron, cfg["train_fraction"], stratify=stratify)
    test, n_enron = drop_near_duplicates(train, test, threshold=threshold)

    sa = load_spamassassin(sa_path)
    sa_test, n_sa = drop_near_duplicates(train, sa, threshold=threshold)

    data = {
        "enron_train": train,
        "enron_test": test,
        "sa_test": sa_test,
        "attacker_ham": sa_test[sa_test.label == 0].reset_index(drop=True),
        "attacker_spam": sa_test[sa_test.label == 1].reset_index(drop=True),
        "dedup": {
            "enron_test_removed": n_enron,
            "sa_removed": n_sa,
            "sa_removed_spam": int(len(sa[sa.label == 1]) - (sa_test.label == 1).sum()),
            "threshold": threshold,
        },
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(data, fh)
    return data


def fit_victim(name: str, data: dict, cfg: dict | None = None, **kwargs):
    """Fit a victim model ("multinomial" or "bernoulli") on enron_train."""
    cfg = cfg or load_config()
    model = MODELS[name](alpha=cfg["smoothing_alpha"], **kwargs)
    train = data["enron_train"]
    return model.fit(train["text"], train["label"])


def describe(df: pd.DataFrame) -> str:
    return (
        f"{len(df):,} messages, {df.label.mean():.1%} spam, "
        f"{df.timestamp.min():%Y-%m-%d} to {df.timestamp.max():%Y-%m-%d}"
    )
