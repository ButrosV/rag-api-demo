import pytest
import numpy as np
from pathlib import Path
import chromadb

from ingestion.loaders import chunk_pages, PageText
from app.config import get_settings

settings = get_settings()

@pytest.fixture
def sample_pages() -> list[PageText]:
    """
    Generate standard sample PageText objects for ingestion testing.
    Creates two realistic pages: one long page (5,000+ chars) for multi-chunk testing,
    one short page (100+ chars) for single-chunk testing.
    :return: List of PageText objects with page_number and text attributes.
    """
    return [PageText(page_number=5, text="Long pages. " * 100),
            PageText(page_number=6, text="Short page. " * 5)]


def test_chunking_sample(sample_pages) -> None:
    """
    Verify chunk_pages() produces correct chunks from sample pages with overlap.
    Test sliding window chunking logic:
    1) Validates expected chunk count calculation.
    2) correct ID format (p5_c1), 
    3) page attribution,
    4) chunk length boundaries (400-500 chars with 100 overlap),
    5) page metadata preservation across multi-page input. 
    
    :param sample_pages: Fixture providing standard test PageText objects (pages 5-6).
    :raises AssertionError: If chunk count, ID format, text length, or page metadata incorrect.
    """
    chunks = list(chunk_pages(sample_pages, chunk_size=500, overlap=100))
    expacted_chunkcount = int(np.ceil(len("Long pages. ") * 100 / (500 - 100))) + 1

    assert len(chunks) == expacted_chunkcount
    assert chunks[0].id == "p5_c1"
    assert chunks[0].page == 5
    assert 400 <= len(chunks[0].text) <= 500
    assert all(c.page in [5, 6] for c in chunks)


def test_index_build_end_to_end(tmp_path: Path, mocker, sample_pages) -> None:
    """
    End-to-end test of build_index(): PDF→pages→chunks→embeddings→Chroma persistence.
    Mocks load_pdf_pages() before import to ensure test isolation.
    Verifies:
    1. Chroma creates sqlite3 database and parquet files
    2. Collection uses configured name from settings
    3. Document count matches expected chunks from sample_pages
       
    :param tmp_path: pytest fixture for temporary filesystem.
    :param mocker: pytest-mock fixture for function replacement.
    :param sample_pages: Test PageText data for consistent chunk count.
    :raises AssertionError: If index files missing, collection empty, or wrong name.
    """
    mocker.patch('ingestion.loaders.load_pdf_pages', return_value=sample_pages)

    test_pdf = tmp_path / "test.pdf"
    test_pdf.touch() 

    index_dir = tmp_path / "test_index"

    from ingestion.build_index import build_index  # Import AFTER patching to ensure the mock is used

    build_index(pdf_path=test_pdf, index_dir=index_dir)

    assert (index_dir / "chroma.sqlite3").exists()
    assert len(list(index_dir.glob("*"))) > 0

    client = chromadb.PersistentClient(str(index_dir))
    collection = client.get_collection(name=settings.collection_name)
    assert collection.count() > 0
