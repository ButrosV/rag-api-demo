## Task 2 – Custom RAG API (Working Notes)

This file summarizes decisions and requirements for implementing **Task 2: Developing a Custom RAG API**.  
You can copy this into a new project folder and use it as the basis for implementation planning.

---

### 1. Goal

Build a **small, self-contained RAG API** that:
- Accepts a user **question**.
- Uses the **NVIDIA document** (provided externally) as the single knowledge source.
- Returns:
  - A **generated answer** grounded in that document.
  - The **relevant text chunks** from the document that were used to generate the answer.

No UI is required; focus is on the API and the RAG logic.

---

### 2. Proposed tech stack

You are free to choose, but a **simple, pragmatic stack** that fits your skills and environment:

- **Language & runtime**
  - Python **3.13** (via conda or venv).

- **Web framework**
  - **FastAPI** (async, good docs, easy to test).
  - **Uvicorn** as the ASGI server.

- **LLM & embeddings**
  - **OpenAI** or **Azure OpenAI**:
    - Chat model: e.g. `gpt-4.1-mini` / `gpt-4o-mini` or close equivalent.
    - Embedding model: e.g. `text-embedding-3-small` (or Azure equivalent).

- **Vector store / similarity search**
  - In-process **Chroma** or **FAISS** (no external infra).

- **Document processing**
  - **PDF loader** to handle “NVIDIA document”:
    - `pypdf`, or
    - `langchain-community` PDF loader.

- **Testing**
  - `pytest` + FastAPI `TestClient` or `httpx`.

- **Config & quality tools (optional but recommended)**
  - `python-dotenv` for API keys & config.
  - `black`, `ruff` for formatting/linting.

---

### 3. API contract (first draft)

- **Endpoint**
  - `POST /ask`

- **Request body**
  - JSON:
    ```json
    {
      "question": "What is X in the NVIDIA document?"
    }
    ```

- **Response body**
  - JSON (example shape):
    ```json
    {
      "answer": "Natural language answer grounded in the NVIDIA document...",
      "contexts": [
        {
          "id": "chunk-001",
          "text": "Relevant excerpt from the document...",
          "page": 5,
          "score": 0.89
        },
        {
          "id": "chunk-002",
          "text": "Another relevant excerpt...",
          "page": 6,
          "score": 0.82
        }
      ]
    }
    ```

You can extend this later with timing, token usage, or debugging info, but keep the initial version minimal.

---

### 4. RAG pipeline – high-level design

#### 4.1. Ingestion & index build (offline step)

1. **Load NVIDIA document**
   - Use a PDF loader to extract text per page or section.

2. **Chunking**
   - Split text into overlapping chunks, for example:
     - Chunk size: **500–1000 characters** (or tokens).
     - Overlap: **100–200 characters**.
   - Store metadata:
     - `id` (e.g., `chunk-001`),
     - `text`,
     - `page` / section label.

3. **Embeddings & vector store**
   - For each chunk:
     - Call the embedding model to get a vector.
   - Store vectors and metadata in **Chroma** or **FAISS**.
   - Save index to disk (e.g., a local folder `./index/`).

4. **Build script**
   - A script like `build_index.py` that:
     - Reads local NVIDIA PDF.
     - Chunks and embeds it.
     - Creates/persists the vector store.
   - This is run manually before starting the API.

#### 4.2. Retrieval + generation (online step)

For each incoming `question`:

1. **Question embedding**
   - Compute embedding with the same embedding model used for chunks.

2. **Vector search**
   - Query the vector store:
     - `top_k` (e.g., 5–10).
   - Get back a ranked list of chunks + scores.

3. **Prompt construction**
   - System prompt:
     - Instruct the model to **answer only using the provided context** from the NVIDIA document, and to say “I don’t know” if the answer cannot be found.
   - User prompt:
     - Include:
       - The **question**.
       - The concatenated **retrieved chunk texts** (with some structure, e.g., numbered “Context 1, Context 2, …”).

4. **Generation**
   - Call chat completion endpoint with the constructed messages.
   - Get back the answer string.

5. **Response assembly**
   - Return:
     - The **answer**.
     - The list of selected **contexts** (chunk id, text, page, score).

---

### 5. Project structure (suggested)

When you create a new project folder, a simple structure could be:

```text
rag-nvidia-api/
  README.md
  .env                # API keys, config (not checked in)
  pyproject.toml      # or requirements.txt
  app/
    main.py           # FastAPI app, /ask endpoint
    rag_pipeline.py   # Retrieval + generation functions
    models.py         # Pydantic request/response models
    config.py         # Settings, env loading
  ingestion/
    build_index.py    # Offline index builder
    loaders.py        # PDF loading & chunking utilities
  tests/
    test_ingestion.py
    test_rag_pipeline.py
    test_api.py
```

We can adjust this layout when starting the actual implementation.

---

### 6. Environment & tooling (on Pop!\_OS with miniconda)

- Create a **dedicated environment**:
  - With conda:
    ```bash
    conda create -n rag-nvidia-api python=3.13
    conda activate rag-nvidia-api
    ```
  - Or with venv:
    ```bash
    python3.13 -m venv .venv
    source .venv/bin/activate
    ```

- Install dependencies (example `pip` set):
  ```bash
  pip install fastapi uvicorn[standard] openai chromadb pypdf pytest
  # optional: python-dotenv httpx langchain-community black ruff
  ```

- You do **not** need:
  - `azd`, `az` CLI,
  - Docker (unless you want to containerize later),
  - Azure Search or Storage (unless you explicitly choose them).

Environment complexity stays low: one Python env + OpenAI/Azure OpenAI key.

---

### 7. Testing outline

- **Unit tests**
  - Chunking: given sample text, chunk boundaries and overlaps are as expected.
  - Retrieval: with a small, synthetic index, ensure a simple query returns expected chunk(s).

- **Integration tests**
  - Use FastAPI `TestClient` to call `POST /ask`:
    - Valid question → 200 + non-empty `answer` + non-empty `contexts`.
    - Empty question → 4xx with clear error.
    - Very long or nonsense question → still 200, but answer may say “I don’t know”.

- **Manual validation**
  - Use curl/Postman to send questions:
    - Check that the answer content is clearly grounded in the returned chunks.

---

### 8. Next steps (for the follow-up chat)

When you start the new project and paste this file there, good first steps for us to work on together:

1. **Confirm stack & models**:
   - Decide OpenAI vs Azure OpenAI.
   - Decide embedding & chat model names.
2. **Lock project structure**:
   - Create folders/files per the suggested layout (we can refine them).
3. **Define config shape**:
   - Environment variables and settings (`.env`, `config.py`).
4. **Design detailed ingestion flow**:
   - Exact chunk size/overlap.
   - How to store page numbers/sections.
5. **Implement step-by-step**:
   - `ingestion/build_index.py` first (so the index exists).
   - Then `app/rag_pipeline.py` retrieval+generation.
   - Then `app/main.py` with `/ask`.
   - Finally, tests.

Bring this file into your new project folder, and we can continue by turning it into a concrete implementation plan and then code. 

