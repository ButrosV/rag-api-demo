## NVIDIA RAG API Showcase

A small, self-contained Retrieval-Augmented Generation (RAG) API over NVIDIA's Q2 FY2024 earnings press release, implemented with FastAPI and OpenAI. Focuses on clean architecture, testability, and grounding of answers in source document.

---

### Task, goal & outcome

- **Task**: Build a `POST /ask` API that:
  - Accepts a natural-language question.
  - Retrieves relevant chunks from a pre-built vector index of the NVIDIA document.
  - Calls an LLM to generate an answer strictly grounded in those chunks.
  - Returns both the **answer** and the **supporting contexts** (chunk text, page, scores).
- **Goal**: Demonstrate a production-style RAG pipeline on a small scope, with:
  - Clear separation between ingestion, retrieval, and API layers.
  - Basic tests and config management.
  - Optional notebook for exploration and manual tesing, not for core logic.
- **Observed outcome (from manual eval in `notebooks/01_sanity_checks_and_experiments.ipynb`)**:
  - On a 5-question eval set, answers were **accurate and concise**, including correct extraction of key numbers (e.g. \$13.51B revenue, 70.1% GAAP gross margin).
  - The model correctly answered “**I don’t know**” when the document did not contain the requested information (AMD/Intel competitors).
  - Retrieved contexts were generally relevant; potential risks are mainly due to **PDF table extraction limits** in `pypdf`, not the RAG wiring.

- For more detailed high level solution and result analysis see [Project Canva Slide Deck](https://www.canva.com/design/DAHEGTYU8E4/aYHQRibEwWfAznp5KiQZOg/edit).

---

### Project Structure

```text
rag-api-nwidia/
  README.md
  Makefile               # Common tasks: install, test, ingest, run-api
  pyproject.toml         # Project metadata + dependencies (Python 3.11)
  .env.example            # Template for local secrets/config
  .gitignore              
  TASK2_RAG_API_CHECKLIST.md  # Public implementation checklist (step-by-step todos)

  app/
    main.py               # FastAPI app, /ask endpoint and wiring
    rag_pipeline.py       # Retrieval + generation pipeline
    schemas.py            # Pydantic request/response models
    config.py             # Settings and env loading

  ingestion/
    build_index.py        # Offline: load PDF, chunk, embed, build vector index
    loaders.py            # PDF loading and chunking utilities

  tests/
    test_ingestion.py     # Unit tests for chunking and index build
    test_rag_pipeline.py  # Retrieval + generation tests
    test_api.py           # FastAPI endpoint tests
  tests/data/             # Small eval fixtures
    test_questions.json   # 5 questions + expected answers/contexts
    expected_answers.md   # Human-readable evaluation criteria

  notebooks/
    01_sanity_checks_and_experiments.ipynb  # Optional: ingestion + retrieval + API sanity checks + manual output evaluation

  data/                 # input data (gitignored)
    NVIDIAAn.pdf        # input dataset (gitignored)
```

---

### Environments & Dependencies

- **Python version**: 3.11 (stable, widely supported for libraries used here).
- **Core stack**:
  - `fastapi`, `uvicorn[standard]` – API and ASGI server.
  - `openai` –  LLM (gpt-4.1-mini) + embeddings (text-embedding-3-small).
  - `tenacity` - retry decorators for LLM,
  - `chromadb` – local vector store for similarity search.
  - `pypdf` – NVIDIA PDF parsing.
  - `python-dotenv` – load `.env` into environment.
  - `pytest`, `pytest-mock`, `httpx` – testing.
  - `ruff` – formatting and linting.
- **Dev-only (optional)**:
  - `jupyterlab`, `pandas` – exploratory experiments in `notebooks/`.

Create and activate a Python 3.11 environment (e.g. if with mamba/conda `mamba create -n rag-api python=3.11`), then install from `pyproject.toml`:

```bash
pip install .
# or, if you prefer extras for notebooks:
# pip install ".[dev]"

# Using the Makefile (optional, convenience):
make install           # installs project in editable mode via pyproject.toml
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

1. **Get data**
Create Data directory and get sample NVIDA data from [google drive](https://drive.google.com/file/d/13vW6HoiK40vORnatUa_rOwsYtdtOXRq4/view?usp=drive_link), if not already present (ie git cloned project does not have `data/` directory).
File can be downloaded from goodle drive manually or with `gdown` after `pip install gdown` in project envirinment.

```bash
mkdir data

# (optional) get file via CLI with gdown
pip install gdown
gdown "https://drive.google.com/file/d/13vW6HoiK40vORnatUa_rOwsYtdtOXRq4/view?usp=drive_link" -O data/NVIDIAAn.pdf  
```
2. **Build the vector index (offline ingestion)**

   ```bash
   # direct
   python -m ingestion.build_index

   # or via Makefile
   make ingest            # build index

   # clean and rebuild index directory
   make clean-index       # remove existing index dir
   make ingest            # or: make rebuild-index
   ```

   This loads the NVIDIA PDF, chunks it, computes embeddings, and persists a vector index under `INDEX_DIR` (e.g. `./index`).

3. **Run the API**

   ```bash
   # direct (port 8001 to avoid clashes)
   uvicorn app.main:app --reload --port 8001

   # or via Makefile
   make run-api
   ```

   The API will load config, open the persisted index, and expose:

   - `POST /ask` – main RAG endpoint.

4. **Query the RAG API**

   ```bash
   curl -X POST http://localhost:8001/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What was NVIDIA Q2 revenue?"}'
   ```

   Expected JSON response:

   - `answer`: grounded natural-language answer.
   - `contexts`: list of chunks with `id`, `text`, `page`, and similarity `score`.

5. **Format, lint, and test (local dev helpers)**

   With the included `Makefile` you can run common tasks quickly:

   ```bash
   make format             # ruff format .
   make lint               # ruff check .
   make fix                # ruff check . --fix
   make test               # pytest tests
   make test-ingestion-e2e # RUN_INGESTION_E2E=1 pytest tests/test_ingestion.py -v
   make check              # lint + test
   ```

---

### Notebooks vs. Core Code

- `notebooks/` are **for exploratory analysis only**:
  - Inspect raw PDF text, test different chunk sizes/overlaps.
  - Manually inspect retrieved chunks and debug relevance.
  - Evaluate model performance manually comparing RAG outouts vs test question responses.
- All **production logic** (ingestion, retrieval, API, tests) lives in `.py` modules under `app/`, `ingestion/`, and `tests/`.
- This separation keeps:
  - The core codebase clean and testable.
  - Your experimentation transparent but clearly marked as non-essential.

---

### Testing & checklist

- Run the full test suite:

```bash
pytest -q
```

Conducted tests:

- Ingestion: chunking behavior and index build.
- RAG pipeline: retrieval correctness on a small synthetic index, end-to-end generation.
- API: `POST /ask` validation and basic success/error cases.
- Output content evaluation suite in `tests/data/` (use optional notebook to run and explore):
    - test_questions.json: 5 questions with expected answers/context
    - expected_answers.md: Detailed success criteria per question for human evaluation

- See `TASK2_RAG_API_CHECKLIST.md` for a high-level, step-by-step checklist of the implementation.
