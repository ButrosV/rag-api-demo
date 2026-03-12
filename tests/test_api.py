from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test /health endpoint returns 200 with {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_question() -> None:
    """Test valid RAG question returns 200 with answer + contexts + scores 0.0-1.0."""
    question = {"question": "What was NVIDIA Q2 revenue?"}
    response = client.post("/ask", json=question)
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert len(data["contexts"]) >= 1
    assert 0 <= data["contexts"][0]["score"] <= 1.0


def test_empty_question() -> None:
    """Test empty string question triggers Pydantic 422 validation error."""
    question = {"question": ""}
    response = client.post("/ask", json=question)
    assert response.status_code == 422  # Pydantic min_length=1
    errors = response.json()["detail"]
    assert any("question" in str(err) for err in errors)


def test_missing_question() -> None:
    """Test missing question field triggers Pydantic 422 validation error."""
    response = client.post("/ask", json={})
    assert response.status_code == 422  # Pydantic min_length=1
    assert "question" in str(response.json()["detail"])


def test_response_struct() -> None:
    """Test AskResponse has required fields with correct types/constraints."""
    question = {"question": "What was NVIDIA Q2 revenue?"}
    response = client.post("/ask", json=question)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "contexts" in data
    assert "timestamp" in data
    assert isinstance(data["contexts"], list)
    assert 1 <= len(data["contexts"]) <= 10
