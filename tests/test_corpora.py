"""Loaders for the real corpora, the per-class temporal split and
near-duplicate removal. All data here is synthetic and tiny."""

import pandas as pd
import pytest

from spam_nb.data import (
    drop_near_duplicates,
    load_enron_csv,
    load_spamassassin,
    near_duplicate_mask,
    parse_email,
    temporal_split,
)

# ----------------------------------------------------------------------------
# Enron single-file release
# ----------------------------------------------------------------------------


def test_load_enron_csv(tmp_path):
    path = tmp_path / "enron.csv"
    pd.DataFrame(
        {
            "Message ID": [0, 1, 2],
            "Subject": ["meeting", "cheap pills", None],
            "Message": ["see you at 3", None, "body only"],
            "Spam/Ham": ["ham", "spam", "ham"],
            "Date": ["2001-01-02", "2004-05-06", "2000-12-31"],
        }
    ).to_csv(path, index=False)

    df = load_enron_csv(path)

    assert list(df.columns) == ["text", "label", "timestamp"]
    assert df["label"].tolist() == [0, 0, 1]  # sorted by date
    assert df["text"].tolist() == ["\nbody only", "meeting\nsee you at 3", "cheap pills\n"]


def test_load_enron_csv_rejects_other_files(tmp_path):
    path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Enron-Spam CSV"):
        load_enron_csv(path)


# ----------------------------------------------------------------------------
# Per-class temporal split
# ----------------------------------------------------------------------------


def _skewed_corpus():
    # All ham is older than all spam, as in Enron-Spam.
    ham = pd.DataFrame(
        {"text": [f"ham {i}" for i in range(10)], "label": 0,
         "timestamp": pd.date_range("2000-01-01", periods=10, tz="UTC")}
    )
    spam = pd.DataFrame(
        {"text": [f"spam {i}" for i in range(10)], "label": 1,
         "timestamp": pd.date_range("2004-01-01", periods=10, tz="UTC")}
    )
    return pd.concat([ham, spam], ignore_index=True)


def test_global_split_on_skewed_corpus_leaves_only_spam_in_test():
    _, test = temporal_split(_skewed_corpus(), 0.7)
    assert test["label"].eq(1).all()  # the problem the stratified split fixes


def test_stratified_split_keeps_both_classes_and_chronology():
    train, test = temporal_split(_skewed_corpus(), 0.7, stratify=True)

    assert len(train) == 14 and len(test) == 6
    for label in (0, 1):
        tr = train[train.label == label]
        te = test[test.label == label]
        assert len(tr) == 7 and len(te) == 3
        assert tr["timestamp"].max() <= te["timestamp"].min()
    assert set(train["text"]).isdisjoint(test["text"])


# ----------------------------------------------------------------------------
# SpamAssassin raw messages
# ----------------------------------------------------------------------------

PLAIN = b"""From alice@example.com  Thu Aug 22 12:36:23 2002
Date: Thu, 22 Aug 2002 07:36:16 -0400
Subject: Re: meeting notes
Content-Type: text/plain

See you at the project meeting.
"""

FORGED_DATE_HTML = b"""Received: from relay.example.net by mx.example.org; Fri, 23 Aug 2002 11:17:32 +0000
Date: Tue, 22 Aug 0102 12:07:35 +0800
Subject: WIN CASH
Content-Type: text/html

<html><style>p{color:red}</style><body><p>Claim your <b>free</b> prize</p>
<div style="display:none">meeting budget</div></body></html>
"""

MULTIPART = b"""From bob@example.com  Mon Sep 02 10:00:00 2002
Subject: report
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="XX"

--XX
Content-Type: text/plain

quarterly report attached
--XX
Content-Type: application/pdf
Content-Disposition: attachment; filename="r.pdf"

JVBERi0xLjQK
--XX--
"""


def test_parse_plain_message():
    text, when = parse_email(PLAIN)
    assert text.startswith("Re: meeting notes\n")
    assert "project meeting" in text
    assert when == pd.Timestamp("2002-08-22 12:36:23", tz="UTC")


def test_forged_date_falls_back_to_received_header():
    text, when = parse_email(FORGED_DATE_HTML)
    assert when == pd.Timestamp("2002-08-23 11:17:32", tz="UTC")
    # HTML converted to text; styles dropped; hidden text kept (naive
    # extractor -- the hidden-text attack A5 relies on this).
    assert "free" in text and "prize" in text
    assert "color" not in text
    assert "meeting budget" in text


def test_attachments_ignored():
    text, _ = parse_email(MULTIPART)
    assert "quarterly report" in text
    assert "JVBER" not in text


def test_load_spamassassin_layout(tmp_path):
    for folder, files in {
        "easy_ham": {"00001.a": PLAIN, "cmds": b"mv x y"},
        "spam_2": {"00001.b": FORGED_DATE_HTML, "00001.b.json": b"{}"},
        "hard-ham-1": {"00002.c.txt": MULTIPART},
        "notes": {"readme": b"ignored"},
    }.items():
        (tmp_path / folder).mkdir()
        for name, content in files.items():
            (tmp_path / folder / name).write_bytes(content)

    df = load_spamassassin(tmp_path)

    assert len(df) == 3
    assert df["label"].tolist() == [0, 1, 0]  # chronological
    assert df["group"].tolist() == ["easy_ham", "spam_2", "hard-ham-1"]
    assert df["timestamp"].is_monotonic_increasing


# ----------------------------------------------------------------------------
# Near-duplicates
# ----------------------------------------------------------------------------

BASE = (
    "dear friend we are pleased to inform you that your email address has "
    "won the international lottery draw please contact our claims agent today"
)


def test_near_duplicate_detection():
    reference = [BASE, "completely different text about a project meeting next week"]
    candidates = [
        BASE,                                     # exact copy
        BASE.upper() + " !!!",                    # same tokens after normalising
        BASE.replace("today", "tomorrow"),        # one word changed
        "the quarterly budget review is moved to friday afternoon in room four",
        "",
    ]
    mask = near_duplicate_mask(reference, candidates)
    assert mask.tolist() == [True, True, True, False, False]


def test_threshold_controls_strictness():
    edited = " ".join(BASE.split()[:12]) + " some new words appended here to change it a lot"
    assert not near_duplicate_mask([BASE], [edited], threshold=0.8)[0]
    assert near_duplicate_mask([BASE], [edited], threshold=0.2)[0]


def test_drop_near_duplicates_frame():
    ref = pd.DataFrame({"text": [BASE]})
    cand = pd.DataFrame({"text": [BASE, "unrelated short note"], "label": [1, 0]})
    kept, removed = drop_near_duplicates(ref, cand)
    assert removed == 1
    assert kept["text"].tolist() == ["unrelated short note"]
