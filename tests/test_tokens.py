"""Fixture-only tests for step-10 contextual token counting."""

import re
from importlib import import_module

tokens = import_module("data.build.10_tokens")


class AsciiWhitespaceTokenizer:
    """Tiny predictable tokenizer fixture for ASCII test strings."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        assert add_special_tokens is False
        return re.findall(r"x:|[A-Za-z]+|[^A-Za-z\s]", text)

    def convert_ids_to_tokens(self, token_ids: list[str]) -> list[str]:
        return token_ids


def test_contextual_difference_matches_direct_ascii_counts() -> None:
    tokenizer = AsciiWhitespaceTokenizer()
    prefix = "x:"
    for word in ("cat", "hello", "good dog", "blue house"):
        count, context_tokens = tokens.contextual_tokenization(tokenizer, prefix, word)
        direct = len(tokenizer.encode(word, add_special_tokens=False))
        assert count == direct
        assert count == len(tokenizer.encode(prefix + " " + word, add_special_tokens=False)) - len(
            tokenizer.encode(prefix, add_special_tokens=False)
        )
        assert context_tokens == tokenizer.encode(prefix + " " + word, add_special_tokens=False)[
            len(tokenizer.encode(prefix, add_special_tokens=False)):
        ]
        assert len(context_tokens) == count
