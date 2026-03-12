## Task 2 – Custom RAG API Implementation Checklist

This is a high-level, best-practice roadmap for implementing a small NVIDIA‑document RAG API.  

---

### 1. Clarify scope & success criteria

- [X] **Confirm goal**: Single `/ask` endpoint that answers questions grounded only in the NVIDIA document and returns supporting chunks.
- [X] **Confirm non-goals**: No UI, no multi-tenant auth, no external databases (unless later extended).
- [X] **Define “good enough”**: Agree on a small set of example questions where answers must be accurate and grounded.

---

### 2. Environment & tooling

- [X] **Create Python environment** (conda env or venv, Python 3.13).
- [X] **Add dependency file** (`pyproject.toml` or `requirements.txt`) with FastAPI, Uvicorn, OpenAI SDK, vector store (Chroma/FAISS), PDF loader, pytest.
- [X] **Create `.env.example`** documenting required variables (e.g., `OPENAI_API_KEY`, model names, index path).
- [X] **Install and configure basic tooling**: `black`, `ruff` (optional but recommended).

---

### 3. Project structure & config

- [X] **Create base folders**: `app/`, `ingestion/`, `tests/`.
- [X] **Create core app files**: `app/main.py`, `app/rag_pipeline.py`, `app/models.py`, `app/config.py`.
- [X] **Create ingestion files**: `ingestion/build_index.py`, `ingestion/loaders.py`.
- [X] **Implement `config.py`** to load settings from env (API key, model names, index directory, chunk params).

---

### 4. Choose models & vector store

- [X] **Decide provider**: OpenAI vs Azure OpenAI.
- [X] **Select chat model** (e.g., `gpt-4.1-mini` / `gpt-4o-mini` or equivalent).
- [X] **Select embedding model** (e.g., `text-embedding-3-small` or equivalent).
- [X] **Select vector store** (Chroma or FAISS) and define persistence path (e.g., `./index/`).

---

### 5. Ingestion & index build (offline)

- [X] **Implement PDF loading** in `ingestion/loaders.py`:
  - [X] Load NVIDIA PDF from disk.
  - [X] Extract text per page (and keep page numbers).
- [X] **Implement chunking logic**:
  - [X] Decide chunk size and overlap (e.g., 500–1000 chars, 100–200 overlap).
  - [X] Produce chunks with metadata: `id`, `text`, `page`.
- [X] **Implement embedding + index creation** in `ingestion/build_index.py`:
  - [X] For each chunk, call embedding model.
  - [X] Store vectors + metadata into Chroma.
  - [X] Persist index to disk.
- [X] **Add minimal tests** for ingestion (`tests/test_ingestion.py`):
  - [X] Chunking produces expected number/size of chunks on sample text.
  - [X] Index builds successfully and can be reloaded.
- [X] **Run ingestion locally** to generate the index once.

---

### 6. Retrieval & generation pipeline (online)

- [X] **Implement retrieval function** in `app/rag_pipeline.py`:
  - [X] Load vector store from disk at startup.
  - [X] Given a question, compute embedding and perform `top_k` similarity search.
  - [X] Return ranked chunks with scores and metadata.
- [X] **Design prompts**:
  - [X] System prompt: answer **only** using provided context from NVIDIA document, otherwise say “I don’t know”.
  - [X] User prompt: include the question + structured list of retrieved contexts.
- [X] **Implement generation function**:
  - [X] Call chat model with system + user messages.
  - [X] Return answer string and the contexts used.
- [X] **Add tests** for pipeline (`tests/test_rag_pipeline.py`):
  - [X] With a small synthetic index, verify that a simple query retrieves the right chunk(s).
  - [X] Verify answer is non-empty and refers to retrieved context.

---

### 7. API layer (FastAPI)

- [X] **Define Pydantic models** in `app/models.py`:
  - [X] `AskRequest` with `question: str`.
  - [X] `ContextChunk` with `id`, `text`, `page`, `score` == `RetrievedChunk` schema.
  - [X] `AskResponse` with `answer` and `contexts` and `timestamp`.
- [X] **Create FastAPI app** in `app/main.py`:
  - [X] Initialize config and load vector store on startup.
  - [X] Implement `POST /ask` endpoint that:
    - [X] Validates `question` (non-empty, reasonable length).
    - [X] Calls retrieval + generation.
    - [X] Returns `AskResponse`.
  - [X] Add basic error handling (e.g., missing index, upstream LLM error).
- [X] **Add API tests** in `tests/test_api.py`:
  - [X] `POST /ask` with valid question → 200 + answer + non-empty contexts.
  - [X] Empty question → 4xx with clear error message.

---

### 8. Evaluation & manual validation

- [X] **Prepare a small eval set** of realistic questions about the NVIDIA document.
- [ ] **Run manual checks**:
  - [ ] Answers are correct and clearly supported by returned chunks.
  - [ ] When information is missing from the document, the model says “I don’t know”.
- [ ] **Iterate on chunking & retrieval** parameters if answers are too generic or hallucinated.

---

### 9. Observability & robustness (nice-to-have)

- [ ] **Add basic logging** (question text, latency, number of chunks, truncated answer) with PII-safe practices.
- [ ] **Expose simple health endpoint** (e.g., `GET /health`) to report that index and API key are available.
- [ ] **Add basic rate limiting or guardrails** if needed (e.g., max question length).

---

### 10. Packaging & documentation

- [ ] **Write a short `README.md`**:
  - [ ] Project overview and architecture diagram (optional).
  - [ ] How to set up env, run ingestion, start API, and run tests.
  - [ ] Example `curl`/HTTP request to `/ask`.
- [ ] **Add `TASK2_RAG_API_NOTES.md` link** from README for deeper design notes.
- [ ] **Optionally add a `Makefile`** or scripts for common tasks (`make ingest`, `make run-api`, `make test`).
