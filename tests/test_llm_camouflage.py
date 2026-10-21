import pandas as pd
import pytest

from spam_nb.llm_camouflage import (
    PROMPTS,
    clean_response,
    defang,
    evasion_attribution,
    has_live_contact,
    is_refusal,
    placeholders_kept,
    rule_based_rewrite,
    sample_spam,
    token_change_rate,
)
from spam_nb.naive_bayes import BernoulliNB, MultinomialNB

SPAM = (
    "CONGRATULATIONS!!! You are a WINNER. Click here: http://bad-site.example.com/claim?id=7 "
    "or visit www.cheap-meds.biz today. Email claims@prize-office.net or call "
    "(555) 123-4567 / +1 555.987.6543. Only $49.99, LIMITED TIME offer, 100% free shipping!!!"
)

# ----------------------------------------------------------------------------
# Defanging
# ----------------------------------------------------------------------------


def test_defang_removes_all_live_contacts():
    out = defang(SPAM)
    assert not has_live_contact(out)
    assert out.count("[URL]") == 2
    assert out.count("[EMAIL]") == 1
    assert out.count("[PHONE]") == 2


def test_defang_keeps_the_pitch():
    out = defang(SPAM)
    for phrase in ("WINNER", "Click here", "$49.99", "LIMITED TIME", "free shipping"):
        assert phrase in out


@pytest.mark.parametrize("text", [
    "see https://x.org/a", "go to example.com now", "ftp://files.example.net/x",
    "mail me: a.b+c@d-e.co.uk", "call 555-123-4567",
])
def test_defang_formats(text):
    assert not has_live_contact(defang(text))


def test_defang_leaves_plain_text_alone():
    text = "The meeting is at 3.30 in room 12; budget is 1,200 dollars."
    assert defang(text) == text


def test_placeholders_kept():
    original = "Visit [URL] or write to [EMAIL]."
    assert placeholders_kept(original, "Please see [URL]; reply to [EMAIL].")
    assert not placeholders_kept(original, "Please see the site.")
    assert placeholders_kept("no contact", "anything")


# ----------------------------------------------------------------------------
# LLM response handling
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "I'm sorry, but I can't help with that.",
    "I cannot assist with rewriting spam.",
    "As an AI, I must decline.",
    "Sorry, I won't do this.",
    "   ",
])
def test_refusals_detected(text):
    assert is_refusal(text)


def test_normal_rewrite_not_a_refusal():
    assert not is_refusal("Hi Sam,\n\nI wanted to let you know about an offer at [URL].")


def test_clean_response_strips_wrappers():
    raw = "Here is the rewritten email:\n\nHi Sam, details at [URL]."
    assert clean_response(raw) == "Hi Sam, details at [URL]."
    assert clean_response("```\nHi [URL]\n```") == "Hi [URL]"


def test_prompts_contain_message_slot():
    for prompt in PROMPTS.values():
        assert "{message}" in prompt
        assert prompt.format(message="X").endswith("X")


# ----------------------------------------------------------------------------
# Rule-based fallback
# ----------------------------------------------------------------------------


def test_rule_based_rewrite_softens_and_keeps_placeholders():
    original = defang(SPAM)
    out = rule_based_rewrite(original, "paraphrase", seed=1)
    assert placeholders_kept(original, out)
    assert not has_live_contact(out)
    assert "!!!" not in out
    assert "WINNER" not in out and "LIMITED" not in out
    assert "$49.99" in out  # the offer is kept


def test_rule_based_rewrite_deterministic_and_padded():
    original = defang(SPAM)
    a = rule_based_rewrite(original, "paraphrase", seed=3)
    assert a == rule_based_rewrite(original, "paraphrase", seed=3)
    padded = rule_based_rewrite(original, "paraphrase_pad", seed=3)
    assert len(padded) > len(a)
    with pytest.raises(ValueError):
        rule_based_rewrite(original, "nonsense")


# ----------------------------------------------------------------------------
# Sampling and analysis
# ----------------------------------------------------------------------------

TRAIN = [
    "meeting budget project review notes", "team lunch tomorrow office",
    "please review the attached report", "budget meeting moved to friday",
    "free prize claim winner click", "cheap pills discount offer now",
    "winner free money claim today", "discount offer click free cash",
]
LABELS = [0, 0, 0, 0, 1, 1, 1, 1]


def test_sample_spam_only_flagged_and_defanged():
    models = [MultinomialNB().fit(TRAIN, LABELS), BernoulliNB().fit(TRAIN, LABELS)]
    test = pd.DataFrame({
        "text": [
            "free prize winner click http://x.com " * 3,
            "cheap pills discount offer call 555-123-4567 " * 3,
            "meeting budget project review notes team lunch " * 3,  # spam-labelled but hammy
            "budget meeting tomorrow " * 3,
        ],
        "label": [1, 1, 1, 0],
    })
    sample = sample_spam(test, models, n=2, min_tokens=5)
    assert len(sample) == 2
    assert all(not has_live_contact(t) for t in sample.original)
    assert all(m.margin(sample.original.tolist()).min() > 0 for m in models)
    with pytest.raises(ValueError):
        sample_spam(test, models, n=3, min_tokens=5)


def test_token_change_rate():
    assert token_change_rate("a b c", "a b c") == 0.0
    assert token_change_rate("a b c", "a x c") == pytest.approx(1 / 3)
    assert token_change_rate("", "") == 0.0
    assert token_change_rate("a b", "") == 1.0


def test_evasion_attribution_finds_removed_spam_words():
    model = MultinomialNB().fit(TRAIN, LABELS)
    result = evasion_attribution(
        model, ["free prize winner click"], ["prize meeting budget"], top=5
    )
    removed = dict(result["removed_spam_evidence"])
    added = dict(result["added_ham_evidence"])
    assert {"free", "winner", "click"} <= set(removed)
    assert {"meeting", "budget"} <= set(added)
