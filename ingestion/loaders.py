from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageText:
    page_number: int
    text: str


def load_pdf_pages(path: Path) -> Iterable[PageText]:
    # Placeholder implementation to be filled in later.
    raise NotImplementedError
