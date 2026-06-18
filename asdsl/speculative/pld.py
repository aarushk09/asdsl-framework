"""Prompt Lookup Decoding (PLD) — lossless n-gram draft for greedy verify."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PLDConfig:
    max_ngram: int = 3
    min_ngram: int = 2
    max_draft_k: int = 6


class PromptLookupDecoder:
    """Match prior context n-grams; propose continuation tokens without a draft model."""

    def __init__(self, config: PLDConfig | None = None) -> None:
        self.config = config or PLDConfig()

    def lookup(self, context: list[int]) -> list[int]:
        """Return draft token ids (may be empty)."""
        if len(context) < self.config.min_ngram:
            return []
        for n in range(self.config.max_ngram, self.config.min_ngram - 1, -1):
            if len(context) < n:
                continue
            suffix = tuple(context[-n:])
            for i in range(len(context) - n):
                if tuple(context[i : i + n]) != suffix:
                    continue
                start = i + n
                if start >= len(context):
                    continue
                draft: list[int] = []
                j = start
                while j < len(context) and len(draft) < self.config.max_draft_k:
                    draft.append(int(context[j]))
                    j += 1
                if draft:
                    return draft
        return []
