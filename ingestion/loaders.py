from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    start_char: int
    length: int


def load_pdf_pages(path: Path) -> Iterable[PageText]:
    """
    Load PDF file and yield text content from each page as PageText objects.
    Processes PDF pages lazily (one at a time) to minimize memory usage for large documents.
    :param path: Path to the PDF file.
    :return: Iterable of PageText objects (page_number, text) for each page.
    :raises FileNotFoundError: If PDF file does not exist at the specified path.
    """
    logging.info(f"Loading Data PDF from {path}.")
    if not path.is_file():
        logging.error(f"Data PDF missing at {path}")
        raise FileNotFoundError(f"Data PDF not found at path: {path}")

    with open(path, "rb") as file:
        reader = PdfReader(file)
        logging.info(f"Data PDF loaded. Total pages: {len(reader.pages)}.")
        for page_nr, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            yield PageText(page_number=page_nr, text=text)


def load_csv(path: Path) -> Iterable[dict]:
    raise NotImplementedError("CSV loading not implemented yet.")


def chunk_pages(pages: Iterable[PageText], chunk_size: int = 800, overlap: int = 150) -> Iterable[Chunk]:
    """
    Split PDF pages into overlapping character-based chunks for RAG indexing.

    Use sliding window with configurable size and overlap. Generates page-wise unique IDs
    (p1_c1, p1_c2, p2_c1). Skip empty pages with warning.

    :param pages: Iterable of PageText objects from load_pdf_pages().
    :param chunk_size: Target characters per chunk (default: 800).
    :param overlap: Character overlap between consecutive chunks (default: 150).
    :return: Iterable of Chunk objects with id, text, page, start_char, and length metadata.
    """
    step_size = chunk_size - overlap
    for page in pages:
        if not page.text.strip():
            logging.warning(f"Page {page.page_number} is empty. Skipping.")
            continue

        text = page.text
        chunk_id = 0
        for chunk_start in range(0, len(text), step_size):
            chunk_end = min(chunk_start + chunk_size, len(text))
            chunk_text = text[chunk_start:chunk_end]
            chunk_id += 1
            yield Chunk(
                id=f"p{page.page_number}_c{chunk_id}",
                text=chunk_text,
                page=page.page_number,
                start_char=chunk_start,
                length=len(chunk_text),
            )
