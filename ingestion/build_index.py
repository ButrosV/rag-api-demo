from pathlib import Path
from typing import List
import logging
import chromadb
from openai import OpenAI

from ingestion.loaders import load_pdf_pages, chunk_pages, Chunk
from app.config import get_settings

settings = get_settings()


def embedd_chunks(chunks: List[Chunk], model: str | None = None) -> List[List[float]]:
    """
    Generate OpenAI embeddings for document chunks using batch API.
    Extract text from chunks, calls OpenAI embeddings endpoint, return embedded vectors.
    Use config embedding model by default or provided model override.

    :param chunks: List of Chunk objects from chunk_pages().
    :param model: OpenAI embedding model (e.g. 'text-embedding-3-small').
                  Defaults to settings.embedding_model.
    :return: List of embedding vectors (ie 1536 floats each in
                                    case of `text-embedding-3-small` model).
    """
    model = model or settings.openai_embedding_model
    client = OpenAI(api_key=settings.openai_api_key)
    texts = [chunk.text for chunk in chunks]
    response = client.embeddings.create(input=texts, model=model)
    return [emb.embedding for emb in response.data]


def build_index(pdf_path: Path, index_dir: Path | None = None) -> None:
    """
    Build Chroma vector index from PDF document for RAG retrieval.
    Load PDF → chunks text → generate OpenAI embeddings
                → store in persistent Chroma collection.
    Replace any existing data in the collection. Uses settings from app.config.

    :param pdf_path: Path to input PDF file (e.g. data/NVIDIAAn.pdf).
    :param index_dir: Chroma persistence directory. Defaults to settings.index_dir.
    :return: None (saves index to disk).
    """
    index_dir = index_dir or settings.index_dir
    logging.info(f"Loading {pdf_path}...")
    pages = list(load_pdf_pages(pdf_path))
    chunks = list(chunk_pages(pages))
    logging.info(f"Loaded {len(pages)} pages and generated {len(chunks)} chunks")

    embeddings = embedd_chunks(chunks)
    logging.info("Finished embeddings")

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(name=settings.collection_name, metadata={"hnsw:space": "cosine"})

    ids = [chunk.id for chunk in chunks]
    metadatas = [{"page": chunk.page, "start_char": chunk.start_char, "length": chunk.length} for chunk in chunks]
    documents = [chunk.text for chunk in chunks]

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    logging.info(f"Index built: {len(chunks)} chunks in {index_dir}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    pdf_path = PROJECT_ROOT / "data" / "NVIDIAAn.pdf"
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building index from: {pdf_path.resolve()}")
    print(f"Index storage: {index_dir.resolve()}")

    try:
        build_index(pdf_path, index_dir)
        print("Direct run done")
    except Exception as e:
        print(f"This thing failed: {e}")
