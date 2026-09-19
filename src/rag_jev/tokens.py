"""Explicit passage-text budget estimator; excludes prompts and answer tokens."""

from functools import lru_cache

import tiktoken

ESTIMATOR = "cl100k_base passage-text estimate; excludes prompts, separators and output"


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))
