## Task 2 – Custom RAG API Implementation Checklist

This is a high-level, best-practice roadmap for implementing a small NVIDIA‑document RAG API.  
Work through the sections in order; each item is a concrete todo.

---

### 1. Clarify scope & success criteria

- [X] **Confirm goal**: Single `/ask` endpoint that answers questions grounded only in the NVIDIA document and returns supporting chunks.
- [X] **Confirm non-goals**: No UI, no multi-tenant auth, no external databases (unless later extended).
- [ ] **Define “good enough”**: Agree on a small set of example questions where answers must be accurate and grounded.

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
- [ ] **Implement `config.py`** to load settings from env (API key, model names, index directory, chunk params).

---

### 4. Choose models & vector store

- [ ] **Decide provider**: OpenAI vs Azure OpenAI.
- [ ] **Select chat model** (e.g., `gpt-4.1-mini` / `gpt-4o-mini` or equivalent).
- [ ] **Select embedding model** (e.g., `text-embedding-3-small` or equivalent).
- [ ] **Select vector store** (Chroma or FAISS) and define persistence path (e.g., `./index/`).

---

### 5. Ingestion & index build (offline)

- [ ] **Implement PDF loading** in `ingestion/loaders.py`:
  - [ ] Load NVIDIA PDF from disk.
  - [ ] Extract text per page (and keep page numbers).
- [ ] **Implement chunking logic**:
  - [ ] Decide chunk size and overlap (e.g., 500–1000 chars, 100–200 overlap).
  - [ ] Produce chunks with metadata: `id`, `text`, `page`.
- [ ] **Implement embedding + index creation** in `ingestion/build_index.py`:
  - [ ] For each chunk, call embedding model.
  - [ ] Store vectors + metadata into Chroma/FAISS.
  - [ ] Persist index to disk.
- [ ] **Add minimal tests** for ingestion (`tests/test_ingestion.py`):
  - [ ] Chunking produces expected number/size of chunks on sample text.
  - [ ] Index builds successfully and can be reloaded.
- [ ] **Run ingestion locally** to generate the index once.

---

### 6. Retrieval & generation pipeline (online)

- [ ] **Implement retrieval function** in `app/rag_pipeline.py`:
  - [ ] Load vector store from disk at startup.
  - [ ] Given a question, compute embedding and perform `top_k` similarity search.
  - [ ] Return ranked chunks with scores and metadata.
- [ ] **Design prompts**:
  - [ ] System prompt: answer **only** using provided context from NVIDIA document, otherwise say “I don’t know”.
  - [ ] User prompt: include the question + structured list of retrieved contexts.
- [ ] **Implement generation function**:
  - [ ] Call chat model with system + user messages.
  - [ ] Return answer string and the contexts used.
- [ ] **Add tests** for pipeline (`tests/test_rag_pipeline.py`):
  - [ ] With a small synthetic index, verify that a simple query retrieves the right chunk(s).
  - [ ] Verify answer is non-empty and refers to retrieved context.

---

### 7. API layer (FastAPI)

- [ ] **Define Pydantic models** in `app/models.py`:
  - [ ] `AskRequest` with `question: str`.
  - [ ] `ContextChunk` with `id`, `text`, `page`, `score`.
  - [ ] `AskResponse` with `answer: str` and `contexts: list[ContextChunk]`.
- [ ] **Create FastAPI app** in `app/main.py`:
  - [ ] Initialize config and load vector store on startup.
  - [ ] Implement `POST /ask` endpoint that:
    - [ ] Validates `question` (non-empty, reasonable length).
    - [ ] Calls retrieval + generation.
    - [ ] Returns `AskResponse`.
  - [ ] Add basic error handling (e.g., missing index, upstream LLM error).
- [ ] **Add API tests** in `tests/test_api.py`:
  - [ ] `POST /ask` with valid question → 200 + answer + non-empty contexts.
  - [ ] Empty question → 4xx with clear error message.

---

### 8. Evaluation & manual validation

- [ ] **Prepare a small eval set** of realistic questions about the NVIDIA document.
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

