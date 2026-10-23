"""Encoding attacks A5/A6 and the D1 input sanitiser."""

from __future__ import annotations

import html.parser
import re
import unicodedata
from typing import Iterable

from spam_nb.attacks import Attack


def hidden_html_padding(words: str, style: str = "display:none") -> Attack:
    """A5: append text in a non-visible HTML element."""
    if not words.strip():
        raise ValueError("words must not be empty.")
    if not style.strip():
        raise ValueError("style must not be empty.")
    payload = f'<span style="{style}">{words}</span>'
    return lambda messages: [f"{message}\n{payload}" for message in messages]


_HOMOGLYPHS = {
    "a": "а",  # Cyrillic small a
    "c": "с",  # Cyrillic small es
    "e": "е",  # Cyrillic small ie
    "o": "о",  # Cyrillic small o
    "p": "р",  # Cyrillic small er
    "x": "х",  # Cyrillic small ha
    "y": "у",  # Cyrillic small u
}
_REVERSE_HOMOGLYPHS = {value: key for key, value in _HOMOGLYPHS.items()}
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}


def homoglyph_substitution(text: str, fraction: float = 1.0) -> str:
    """A6: replace letters in spammy tokens with Unicode homoglyphs."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    output = []
    replaced = 0
    for character in text:
        if character in _HOMOGLYPHS and replaced / max(1, len(text)) < fraction:
            output.append(_HOMOGLYPHS[character])
            replaced += 1
        else:
            output.append(character)
    return "".join(output)


def zero_width_insertion(text: str) -> str:
    """A6: insert a zero-width separator into each alphabetic token."""
    return re.sub(r"(?<=[A-Za-z])(?=[A-Za-z])", "\u200b", text)


class _VisibleText(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = (
            tag.lower() in {"script", "style", "template", "noscript"}
            or "hidden" in attrs
            or attrs.get("aria-hidden", "").lower() == "true"
            or "display:none" in attrs.get("style", "").replace(" ", "").lower()
        )
        if hidden:
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if self.hidden_depth and tag.lower() in {
            "script", "style", "template", "noscript", "span", "div", "p"
        }:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if not self.hidden_depth:
            self.parts.append(data)


def sanitize_text(text: str) -> str:
    """D1: normalize Unicode, remove hidden HTML and restore confusable text."""
    parser = _VisibleText()
    parser.feed(unicodedata.normalize("NFKC", str(text)))
    visible = "".join(parser.parts)
    visible = "".join(_REVERSE_HOMOGLYPHS.get(char, char) for char in visible)
    visible = "".join(
        char for char in visible
        if char not in _ZERO_WIDTH and not unicodedata.category(char).startswith("C")
    )
    return visible


def sanitize_documents(documents: Iterable[str]) -> list[str]:
    """Apply D1 to a collection of documents."""
    return [sanitize_text(document) for document in documents]