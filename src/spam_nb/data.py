"""Corpus loading, temporal splitting and near-duplicate removal.

Every loader returns a DataFrame with the same three columns, sorted by
time: text (str), label (int, spam = 1, ham = 0), timestamp (UTC).
"""

from __future__ import annotations

import email
import email.policy
import email.utils
import html
import re
import zlib
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from spam_nb.tokenizer import tokenize

# ----------------------------------------------------------------------------
# Common validation
# ----------------------------------------------------------------------------


def standardize(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Validate a raw frame and return the project's standard columns."""
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
    # (pandas >= 3 stores text as a string dtype, not object.)
    if not pd.api.types.is_numeric_dtype(df["label"]):
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


def load_enron_spam(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Load and validate a CSV that already has text/label/timestamp."""
    return standardize(pd.read_csv(path), text_col, label_col, timestamp_col)


def load_enron_csv(path: str | Path) -> pd.DataFrame:
    """Load the single-file Enron-Spam release (MWiechmann/enron_spam_data).

    Columns there: Message ID, Subject, Message, Spam/Ham, Date. The date is
    the arrival date encoded in the original Metsis et al. file names. The
    subject is prepended to the body because both carry spam evidence.
    Ties on the same day keep the release's original row order.
    """
    raw = pd.read_csv(path)
    expected = {"Subject", "Message", "Spam/Ham", "Date"}
    if not expected <= set(raw.columns):
        raise ValueError(
            f"Not the Enron-Spam CSV: expected columns {sorted(expected)}, "
            f"got {list(raw.columns)}"
        )
    frame = pd.DataFrame(
        {
            "text": raw["Subject"].fillna("").astype(str)
            + "\n"
            + raw["Message"].fillna("").astype(str),
            "label": raw["Spam/Ham"],
            "timestamp": raw["Date"],
        }
    )
    return standardize(frame)


# ----------------------------------------------------------------------------
# SpamAssassin public corpus (raw RFC 822 messages)
# ----------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Collect all text in an HTML body (naive extractor, no rendering).

    This deliberately keeps text a reader cannot see (display:none, tiny
    fonts): that is the baseline the hidden-text attack A5 exploits and the
    render-aware sanitiser D1 will remove.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # malformed HTML: fall back to stripping tags
        return html.unescape(re.sub(r"<[^>]+>", " ", markup))
    return " ".join(parser.parts)


def _decode_part(part) -> str:
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        content = payload.decode("latin-1", errors="replace")
    return content if isinstance(content, str) else ""


def parse_email(raw: bytes) -> tuple[str, pd.Timestamp | None]:
    """Return (subject + body text, arrival time) for one raw message.

    Arrival time is the delivery stamp (see _received_time), falling back
    to the Date header only when its year is plausible (1995-2006).

    Plain-text parts are used when present; otherwise HTML parts are
    converted to text. Attachments are ignored.
    """
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    subject = str(msg.get("subject", "") or "")

    plain, markup = [], []
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_decode_part(part))
        elif ctype == "text/html":
            markup.append(_decode_part(part))

    body = "\n".join(plain) if plain else "\n".join(html_to_text(m) for m in markup)

    when = _received_time(raw)
    if when is None:
        try:
            date_header = msg.get("date")
            if date_header:
                when = _to_utc(email.utils.parsedate_to_datetime(str(date_header)))
        except Exception:
            when = None
        if not _plausible(when):
            when = None

    return subject + "\n" + body, when


def _plausible(when) -> bool:
    return when is not None and 1995 <= when.year <= 2006


def _to_utc(dt) -> pd.Timestamp:
    when = pd.Timestamp(dt)
    return when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")


def _received_time(raw: bytes) -> pd.Timestamp | None:
    """Delivery time: the mbox 'From ' line, else the newest Received header.

    Spammers forge the Date header (years like 0102 are common), so the
    time the receiving server stamped is the trustworthy arrival time.
    """
    first = raw.split(b"\n", 1)[0].decode("latin-1", errors="replace")
    if first.startswith("From "):
        try:
            when = pd.Timestamp(" ".join(first.split()[2:]), tz="UTC")
            if _plausible(when):
                return when
        except Exception:
            pass

    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    for header in msg.get_all("Received") or []:  # newest first
        stamp = str(header).rsplit(";", 1)[-1].strip()
        try:
            when = _to_utc(email.utils.parsedate_to_datetime(stamp))
            if _plausible(when):
                return when
        except Exception:
            continue
    return None


def load_spamassassin(root: str | Path) -> pd.DataFrame:
    """Load the SpamAssassin public corpus from an extracted directory.

    Works with the official tarball layout (easy_ham/, hard_ham/, spam/,
    spam_2/, ...) and with mirrors that use names like easy-ham-1/ and add
    .json sidecars. A message is spam when its folder name contains 'spam'.
    Adds columns: group (folder name) and file (file name).
    """
    root = Path(root)
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name.lower()
        if not ("ham" in name or "spam" in name):
            continue
        label = 1 if "spam" in name else 0
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name == "cmds" or path.suffix == ".json":
                continue
            text, when = parse_email(path.read_bytes())
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "timestamp": when,
                    "group": folder.name,
                    "file": path.name,
                }
            )

    if not rows:
        raise ValueError(f"No SpamAssassin messages found under {root}")

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    n_missing = int(frame["timestamp"].isna().sum())
    if n_missing:
        # A handful of messages have unparseable dates; keep them, dated to
        # the earliest valid date so they can never leak into a "future" set.
        frame["timestamp"] = frame["timestamp"].fillna(frame["timestamp"].min())
    # Sort first; standardize() then keeps this order (its sort is stable)
    # and the metadata columns line up row for row.
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    out = standardize(frame)
    out["group"] = frame["group"].to_numpy()
    out["file"] = frame["file"].to_numpy()
    return out


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: earlier emails train, later emails test.

    stratify=False splits the whole corpus at one cut date.

    stratify=True splits each class at its own cut date (oldest
    train_fraction of ham + oldest train_fraction of spam). Use this for
    Enron-Spam, where all ham predates most spam: a single cut leaves a
    test set that is 100% spam, so precision and FPR become meaningless.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    if len(df) < 2:
        raise ValueError("At least two emails are required.")

    if stratify:
        parts = [
            temporal_split(group, train_fraction, stratify=False)
            for _, group in df.groupby("label", sort=True)
        ]
        train = pd.concat([p[0] for p in parts]).sort_values("timestamp", kind="mergesort")
        test = pd.concat([p[1] for p in parts]).sort_values("timestamp", kind="mergesort")
        return train.reset_index(drop=True), test.reset_index(drop=True)

    ordered = df.sort_values("timestamp", kind="mergesort")
    ordered = ordered.reset_index(drop=True)

    split_idx = int(len(ordered) * train_fraction)
    split_idx = max(1, min(split_idx, len(ordered) - 1))

    train = ordered.iloc[:split_idx].copy()
    test = ordered.iloc[split_idx:].copy()

    assert train["timestamp"].max() <= test["timestamp"].min()

    return train, test


# ----------------------------------------------------------------------------
# Near-duplicate removal (shingle MinHash + exact Jaccard check)
# ----------------------------------------------------------------------------

# Universal hashing h(x) = (a*x + b) mod p with p = 2^31 - 1 (prime).
# x, a, b < p, so a*x + b < 2^62 never overflows uint64, and the products
# do wrap modulo p, which is what makes each h a different permutation.
_PRIME = np.uint64((1 << 31) - 1)


def _shingles(text: str, k: int) -> np.ndarray:
    tokens = tokenize(text)
    if len(tokens) < k:
        grams = [" ".join(tokens)] if tokens else []
    else:
        grams = [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]
    hashes = np.fromiter(
        (zlib.crc32(g.encode()) for g in grams), dtype=np.uint64, count=len(grams)
    )
    return np.unique(hashes % _PRIME)


def _minhash(shingles: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if shingles.size == 0:
        return np.full(a.shape, np.iinfo(np.uint64).max, dtype=np.uint64)
    return ((a[:, None] * shingles[None, :] + b[:, None]) % _PRIME).min(axis=1)


def _choose_bands(threshold: float, num_perm: int) -> int:
    """Most selective banding whose LSH threshold (1/b)^(1/r) is well below
    `threshold`, so true near-duplicates are almost never missed."""
    best = num_perm  # r = 1: every shared minhash makes a candidate
    for bands in sorted(d for d in range(1, num_perm + 1) if num_perm % d == 0):
        rows = num_perm // bands
        if (1 / bands) ** (1 / rows) <= 0.7 * threshold:
            best = bands
            break
    return best


def near_duplicate_mask(
    reference: list[str],
    candidates: list[str],
    threshold: float = 0.8,
    k: int = 5,
    num_perm: int = 64,
    bands: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """True for each candidate whose k-word-shingle Jaccard similarity with
    some reference document is >= threshold.

    MinHash with banding finds candidate pairs; each pair is then confirmed
    with the exact Jaccard similarity, so there are no false positives.
    bands=None picks a banding suited to the threshold.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")
    bands = bands or _choose_bands(threshold, num_perm)
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands.")
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(_PRIME), num_perm, dtype=np.uint64)
    b = rng.integers(0, int(_PRIME), num_perm, dtype=np.uint64)
    rows = num_perm // bands

    ref_sh = [_shingles(t, k) for t in reference]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, sh in enumerate(ref_sh):
        if sh.size == 0:
            continue
        sig = _minhash(sh, a, b)
        for band in range(bands):
            buckets[(band, sig[band * rows : (band + 1) * rows].tobytes())].append(i)

    mask = np.zeros(len(candidates), dtype=bool)
    for j, text in enumerate(candidates):
        sh = _shingles(text, k)
        if sh.size == 0:
            continue
        sig = _minhash(sh, a, b)
        seen: set[int] = set()
        for band in range(bands):
            for i in buckets.get((band, sig[band * rows : (band + 1) * rows].tobytes()), ()):
                if i in seen:
                    continue
                seen.add(i)
                inter = np.intersect1d(sh, ref_sh[i], assume_unique=True).size
                union = sh.size + ref_sh[i].size - inter
                if union and inter / union >= threshold:
                    mask[j] = True
                    break
            if mask[j]:
                break
    return mask


def drop_near_duplicates(
    reference: pd.DataFrame, candidates: pd.DataFrame, **kwargs
) -> tuple[pd.DataFrame, int]:
    """Remove candidate rows that near-duplicate any reference row.

    Returns (filtered candidates, number removed). Use it between any
    training set and any test set, and between the attacker's ham pool and
    the victim's training data.
    """
    mask = near_duplicate_mask(
        reference["text"].tolist(), candidates["text"].tolist(), **kwargs
    )
    return candidates.loc[~mask].reset_index(drop=True), int(mask.sum())
