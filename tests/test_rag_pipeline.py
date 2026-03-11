from app.rag_pipeline import retrieve, build_prompt, prompt_llm, retrieve_and_answer, RetrievedChunk


def test_retrieve_struct():
    """
    Verify retrieve() returns list of RetrievedChunk objects.
    Test:
    1. Returns list type
    2. Respects top_k limit
    3. All items are RetrievedChunk instances
    :return: None
    """
    question = "What was NVIDIA's total revenue for Q2?"
    top_k = 2
    chunks = retrieve(question, top_k)
    assert isinstance(chunks, list)
    assert len(chunks) <= top_k
    for chunk in chunks:
        assert isinstance(chunk, RetrievedChunk)


def test_build_prompt_struct():
    """
    Verify build_prompt() creates correct OpenAI message format.
    Tests:
    1. Returns 2-message list
    2. First message has 'system' role
    3. Second message has 'user' role with Context: and Question
    :return: None
    """
    chunks = [RetrievedChunk(id="1", text="test", page=5, score=0.9)]
    _, messages = build_prompt("What?", chunks)
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Context:" in messages[1]["content"]
    assert "Question:" in messages[1]["content"]


def test_prompt_llm_responds():
    """
    Verify prompt_llm() generates context-grounded response.
    Tests:
    1. Returns non-empty string
    2. Uses provided context (mentions 'savannah')
    :return: None
    """
    answer = prompt_llm(
        [
            {
                "role": "user",
                "content": """Context: Black Mamba lives in the savannah.
                           Question: Where does the Black Mamba live?""",
            }
        ]
    )
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "savannah" in answer.lower()


def test_retrieve_and_answer_pipeline():
    """
    Verify complete RAG pipeline works end-to-end.
    Tests:
    1. Returns (answer, chunks) tuple
    2. Answer is non-empty string
    3. Chunks list has 1-2 items
    :return: None
    """
    answer, chunks = retrieve_and_answer("revenue?", top_k=2)
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert len(chunks) > 0 and len(chunks) <= 2
