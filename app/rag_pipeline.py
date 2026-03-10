from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: str
    text: str
    page: int | None
    score: float


def retrieve_and_answer(question: str) -> tuple[str, Sequence[RetrievedChunk]]:
    # Placeholder implementation to be filled in later.
    raise NotImplementedError
