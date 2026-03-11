## NVIDIA RAG API Showcase

A small, self-contained Retrieval-Augmented Generation (RAG) API over NVIDIA's Q2 FY2024 earnings press release, implemented with FastAPI and OpenAI. Focuses on clean architecture, testability, and clear grounding of answers in source document.

---

### Task & Goal

- **Task**: Build a `POST /ask` API that:
  - Accepts a natural-language question.
  - Retrieves relevant chunks from a pre-built vector index of the NVIDIA document.
  - Calls an LLM to generate an answer strictly grounded in those chunks.
  - Returns both the **answer** and the **supporting contexts** (chunk text, page, scores).
- **Goal**: Demonstrate a production-style RAG pipeline on a small scope, with:
  - Clear separation between ingestion, retrieval, and API layers.
  - Basic tests and config management.
  - Optional notebooks for exploration, not for core logic.

---

### Project Structure

```text
rag-api-nwidia/
  README.md
  pyproject.toml          # Project metadata + dependencies (Python 3.11)
  .env.example            # Template for local secrets/config
  .gitignore              
  TASK2_RAG_API_NOTES.md  # Design notes and planning
  TASK2_RAG_API_CHECKLIST.md  # Implementation checklist (step-by-step todos)

  app/
    main.py               # FastAPI app, /ask endpoint and wiring
    rag_pipeline.py       # Retrieval + generation pipeline
    models.py             # Pydantic request/response models
    config.py             # Settings and env loading

  ingestion/
    build_index.py        # Offline: load PDF, chunk, embed, build vector index
    loaders.py            # PDF loading and chunking utilities

  tests/
    test_ingestion.py     # Unit tests for chunking and index build
    test_rag_pipeline.py  # Retrieval + generation tests
    test_api.py           # FastAPI endpoint tests
      data/                 # Test data
        test_questions.json   # 5  questions + expected answers
        expected_answers.md   # evaluation criteria

  notebooks/
    01_experiments_ingestion.ipynb   # Optional: EDA on PDF text, chunking trials
    02_experiments_retrieval.ipynb # Optional: manual inspection of retrieved chunks

  data/                 # input data (gitignored)
    NVIDIAAn.pdf        # input dataset
```

---

### Environments & Dependencies

- **Python version**: 3.11 (stable, widely supported for libraries used here).
- **Core stack**:
  - `fastapi`, `uvicorn[standard]` – API and ASGI server.
  - `openai` –  LLM (gpt-4.1-mini) + embeddings (text-embedding-3-small).
  - `chromadb` – local vector store for similarity search.
  - `pypdf` – NVIDIA PDF parsing.
  - `python-dotenv` – load `.env` into environment.
  - `pytest`, `pytest-mock`, `httpx` – testing.
  - `ruff` – formatting and linting.
- **Dev-only (optional)**:
  - `jupyterlab` – exploratory experiments in `notebooks/`.

Create and activate a Python 3.11 environment (e.g. if with mamba/conda `mamba create -n rag-api python=3.11`), then install from `pyproject.toml`:

```bash
pip install .
pip install ".[dev]"   # adds JupyterLab and other dev extras if needed
```

---

### Configuration & Secrets

- Copy `.env.example` → `.env` and fill in real values:

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
INDEX_DIR=./index
LOG_LEVEL=info
```

- `.env` is **git-ignored**, `.env.example` is committed to document required variables.
- `app/config.py` is responsible for loading these settings and exposing them to the rest of the app.

---

### Usage

1. **Build the vector index (offline ingestion)**

   ```bash
   python -m ingestion.build_index
   ```

   This loads the NVIDIA PDF, chunks it, computes embeddings, and persists a vector index under `INDEX_DIR` (e.g. `./index`).

2. **Run the API**

   ```bash
   uvicorn app.main:app --reload
   ```

   The API will load config, open the persisted index, and expose:

   - `POST /ask` – main RAG endpoint.

3. **Query the RAG API**

   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What was NVIDIA Q2 revenue?"}'
   ```

   Expected JSON response:

   - `answer`: grounded natural-language answer.
   - `contexts`: list of chunks with `id`, `text`, `page`, and similarity `score`.

---

### Notebooks vs. Core Code

- `notebooks/` are **for exploratory analysis only**:
  - Inspect raw PDF text, test different chunk sizes/overlaps.
  - Manually inspect retrieved chunks and debug relevance.
- All **production logic** (ingestion, retrieval, API, tests) lives in `.py` modules under `app/`, `ingestion/`, and `tests/`.
- This separation keeps:
  - The core codebase clean and testable.
  - Your experimentation transparent but clearly marked as non-essential.

---

### Testing

- Run the full test suite:

```bash
pytest -q
```

Planned tests:

- Ingestion: chunking behavior and index build.
- RAG pipeline: retrieval correctness on a small synthetic index, end-to-end generation.
- API: `POST /ask` validation and basic success/error cases.
- Output content evaluation suite in `tests/data/`:
    - test_questions.json: 5 questions with expected answers/context
    - expected_answers.md: Detailed success criteria per question for human evaluation
