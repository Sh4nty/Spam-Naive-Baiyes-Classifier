
import re
from typing import List

TOKEN_PATTERN = re.compile(r"\b[a-z0-9]+\b")


def tokenize(text: str) -> List[str]:
    """Lowercase text and return alphanumeric tokens."""
    if text is None:
        return []

    return TOKEN_PATTERN.findall(str(text).lower())

