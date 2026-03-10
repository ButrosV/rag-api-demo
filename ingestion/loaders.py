from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


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
    
    with open(path, 'rb') as file:
        reader = PdfReader(file)
        logging.info(f"Data PDF loaded. Total pages: {len(reader.pages)}.")
        for page_nr, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            yield PageText(page_number=page_nr, text=text)

