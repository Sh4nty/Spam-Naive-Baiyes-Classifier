"""A7: LLM camouflage (threat model T2, no knowledge of the victim).

Pipeline:
  1. sample_spam()       pick flagged spam from the Enron test set
  2. defang()            neutralise URLs and contact details (ethics, sec. 10)
  3. PROMPTS             two rewrite prompts: paraphrase / paraphrase + padding
  4. scripts/generate_a7.py  runs an LLM (Ollama or OpenAI-compatible API)
     rule_based_rewrite()    the no-LLM fallback paraphraser
  5. scripts/evaluate_a7.py  scores rewrites and attributes evasions to tokens

The original messages are defanged *before* scoring as well, so any change
in the victim's score comes from the rewrite, not from the defanging.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from spam_nb.tokenizer import tokenize

# ----------------------------------------------------------------------------
# Defanging
# ----------------------------------------------------------------------------

PLACEHOLDERS = ("[URL]", "[EMAIL]", "[PHONE]")

_URL = re.compile(
    r"(?i)\b(?:(?:https?|ftp)://|www\.)[^\s<>\"')\]]+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|net|org|biz|info|us|uk|ru|cn|de|tv|cc|ws|to)\b(?:/[^\s<>\"')\]]*)?"
)
_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
)


def defang(text: str) -> str:
    """Replace e-mail addresses, URLs and phone numbers with placeholders."""
    text = _EMAIL.sub("[EMAIL]", text)  # before URLs: emails contain domains
    text = _URL.sub("[URL]", text)
    text = _PHONE.sub("[PHONE]", text)
    return text


def placeholders_kept(original: str, rewrite: str) -> bool:
    """The pitch must stay actionable: every placeholder type in the
    original must still appear in the rewrite."""
    return all(p in rewrite for p in PLACEHOLDERS if p in original)


def has_live_contact(text: str) -> bool:
    """True if a rewrite introduced a real URL, e-mail or phone number."""
    return bool(_EMAIL.search(text) or _URL.search(text) or _PHONE.search(text))


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------


def sample_spam(
    test: pd.DataFrame,
    models: Iterable,
    n: int = 300,
    seed: int = 42,
    min_tokens: int = 20,
    max_chars: int = 2000,
) -> pd.DataFrame:
    """Sample n spam messages that every model flags when clean.

    Messages are defanged and truncated to max_chars (LLM cost), and the
    selection is re-checked after that so every sampled original is still
    flagged by every model. Returns columns: id, original.
    """
    spam = test[test.label == 1].copy()
    spam["original"] = [defang(t)[:max_chars] for t in spam.text]
    spam = spam[[len(tokenize(t)) >= min_tokens for t in spam.original]]
    # Drop exact duplicates (spam campaigns repeat messages).
    spam = spam.drop_duplicates("original")
    flagged = np.ones(len(spam), dtype=bool)
    for model in models:
        flagged &= model.margin(spam.original.tolist()) > 0
    spam = spam[flagged]
    if len(spam) < n:
        raise ValueError(f"Only {len(spam)} eligible messages; asked for {n}.")
    picked = spam.sample(n=n, random_state=seed).sort_index()
    return pd.DataFrame({
        "id": [f"enron-test-{i}" for i in picked.index],
        "original": picked.original.tolist(),
    }).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Prompts (proposal: paraphrase only, then paraphrase plus benign padding)
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are helping a university research project that measures how text "
    "classifiers respond to rewritten e-mails. All links and contact details "
    "have already been replaced by placeholders, and nothing will be sent to "
    "anyone. Reply with the rewritten e-mail only: no preface, no notes."
)

PROMPTS = {
    "paraphrase": (
        "Rewrite the e-mail below so it reads like a normal, legitimate "
        "message written by a real person in a calm, professional tone. Keep "
        "the same offer and the same call to action. Keep every placeholder "
        "([URL], [EMAIL], [PHONE]) exactly as written, and do not add any new "
        "links, addresses or phone numbers.\n\nE-mail:\n{message}"
    ),
    "paraphrase_pad": (
        "Rewrite the e-mail below so it reads like a normal, legitimate "
        "message written by a real person in a calm, professional tone. Keep "
        "the same offer and the same call to action. Keep every placeholder "
        "([URL], [EMAIL], [PHONE]) exactly as written, and do not add any new "
        "links, addresses or phone numbers. Then add a short, friendly "
        "paragraph of ordinary everyday news (for example about work, a "
        "meeting, family or the weekend) so that the whole e-mail reads like "
        "routine personal correspondence.\n\nE-mail:\n{message}"
    ),
}

_REFUSAL = re.compile(
    r"(?i)^\s*(?:i\s*(?:'m|am)\s*sorry|i\s*can(?:'|no)t|i\s*(?:won't|will not)|"
    r"as an ai|i'm unable|i am unable|sorry,)"
)


def is_refusal(text: str) -> bool:
    return not text.strip() or bool(_REFUSAL.search(text[:200]))


def clean_response(text: str) -> str:
    """Strip common LLM wrappers: code fences and 'Here is ...:' prefaces."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n|\n```$", "", text)
    text = re.sub(r"(?i)^here(?:'s| is) (?:the |a |your )?rewritten[^\n]*:\s*\n", "", text)
    return text.strip()


# ----------------------------------------------------------------------------
# Rule-based fallback paraphraser (no LLM)
# ----------------------------------------------------------------------------

# Marketing register -> plain register. Applied as whole-word, case-insensitive.
_SUBSTITUTIONS = [
    (r"click here", "see the details here"),
    (r"act now", "let me know when you can"),
    (r"buy now", "you can order it"),
    (r"order now", "you can order it"),
    (r"call now", "feel free to call"),
    (r"limited time( only)?", "for a short while"),
    (r"special offer", "an offer"),
    (r"(100%|absolutely) free", "at no cost"),
    (r"free of charge", "at no cost"),
    (r"free", "complimentary"),
    (r"guarantee(d)?", "promise"),
    (r"amazing|incredible|unbelievable|fantastic", "good"),
    (r"lowest prices?", "good prices"),
    (r"cheap(est)?", "affordable"),
    (r"discount(s|ed)?", "reduced price"),
    (r"save( up to)?", "you could save"),
    (r"winner", "selected participant"),
    (r"congratulations", "good news"),
    (r"dear (friend|sir|madam|customer|user)", "hi"),
    (r"unsubscribe|remove me|opt[- ]out", "reply to stop these notes"),
    (r"money back", "refund"),
    (r"risk[- ]free", "with a refund option"),
    (r"no obligation", "no commitment"),
    (r"offer expires", "this ends"),
    (r"don'?t (miss|delay)", "do consider it"),
    (r"(urgent|important|attention)", "a quick note"),
]
_SUBSTITUTIONS = [(re.compile(rf"(?i)\b(?:{p})\b"), r) for p, r in _SUBSTITUTIONS]

_GREETINGS = ["Hi,", "Hello,", "Hi there,", "Good morning,"]
_SIGNOFFS = ["Best regards,\nSam", "Thanks,\nAlex", "Kind regards,\nJordan", "Cheers,\nChris"]
_PADDING = [
    "On another note, the team meeting has moved to Thursday afternoon, and "
    "I will send round the updated notes and the schedule for next week.",
    "Hope your week is going well. We finally finished the garden over the "
    "weekend, and the kids enjoyed the trip to the lake on Sunday.",
    "Also, thanks again for the report you shared last week; I went through "
    "the figures with the group and we will discuss the budget on Monday.",
    "By the way, the office will be closed on Friday for maintenance, so "
    "please plan any calls for earlier in the week.",
]


def _sentence_case(match: re.Match) -> str:
    word = match.group(0)
    return word.capitalize() if len(word) > 3 else word


def rule_based_rewrite(text: str, variant: str = "paraphrase", seed: int = 0) -> str:
    """Template paraphraser used when no LLM is available.

    Softens the marketing register (substitutions above), removes shouting
    (ALL-CAPS words, runs of '!' and '$'), and frames the message as a
    personal note; 'paraphrase_pad' adds one benign everyday paragraph.
    Placeholders and the offer itself are preserved.
    """
    if variant not in PROMPTS:
        raise ValueError(f"variant must be one of {list(PROMPTS)}")
    rng = np.random.default_rng(seed)
    protected = {p: f"\x00{i}\x00" for i, p in enumerate(PLACEHOLDERS)}
    for p, token in protected.items():
        text = text.replace(p, token)

    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\b[A-Z]{2,}\b", _sentence_case, text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\${2,}", "$", text)
    text = re.sub(r"[*=~_#]{3,}", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

    for p, token in protected.items():
        text = text.replace(token, p)

    parts = [_GREETINGS[rng.integers(len(_GREETINGS))], "", text]
    if variant == "paraphrase_pad":
        parts += ["", _PADDING[rng.integers(len(_PADDING))]]
    parts += ["", _SIGNOFFS[rng.integers(len(_SIGNOFFS))]]
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------


def token_change_rate(original: str, rewrite: str) -> float:
    """Normalised token-level edit distance (Levenshtein / max length)."""
    a, b = tokenize(original), tokenize(rewrite)
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def evasion_attribution(model, originals, rewrites, top: int = 20) -> dict:
    """Which tokens explain the score change across a set of rewrites.

    For each pair, the per-token contribution x_t * w_t before and after is
    compared. 'removed_spam_evidence' sums the drop in positive (spammy)
    contributions; 'added_ham_evidence' sums new negative (hammy) ones.
    """
    removed, added = Counter(), Counter()
    for original, rewrite in zip(originals, rewrites):
        before = {tok: c for tok, _, c in model.token_contributions(original)}
        after = {tok: c for tok, _, c in model.token_contributions(rewrite)}
        for tok in set(before) | set(after):
            delta = after.get(tok, 0.0) - before.get(tok, 0.0)
            if before.get(tok, 0.0) > 0 and delta < 0:
                removed[tok] += -delta
            elif delta < 0:
                added[tok] += -delta
    return {
        "removed_spam_evidence": removed.most_common(top),
        "added_ham_evidence": added.most_common(top),
    }
