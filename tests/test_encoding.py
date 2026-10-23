import pytest

from spam_nb.encoding import (
    hidden_html_padding,
    homoglyph_substitution,
    sanitize_documents,
    sanitize_text,
    zero_width_insertion,
)


def test_a5_hidden_html_padding_appends_non_visible_payload():
    output = hidden_html_padding("free budget")(["original"])[0]
    assert output.startswith("original\n")
    assert 'style="display:none"' in output
    assert "free budget" in output


def test_a6_encoding_attacks_change_text_without_changing_length():
    original = "free offer"
    homoglyphs = homoglyph_substitution(original)
    zero_width = zero_width_insertion(original)
    assert homoglyphs != original
    assert zero_width != original
    assert len(homoglyphs) == len(original)
    assert "\u200b" in zero_width


def test_d1_normalizes_and_restores_obfuscations():
    attacked = homoglyph_substitution(zero_width_insertion("free offer"))
    assert sanitize_text(attacked) == "free offer"


def test_d1_drops_hidden_html_and_controls_but_keeps_visible_text():
    text = '<p>free</p><span style="display: none">offer</span><script>click</script>\x00 budget'
    assert sanitize_text(text) == "free budget"
    assert sanitize_documents([text]) == ["free budget"]


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_homoglyph_substitution_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        homoglyph_substitution("free", fraction)