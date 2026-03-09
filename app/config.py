from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Minimal settings for this RAG API.

    Loaded from environment variables and/or a local `.env` file (see `.env.example`).
    Keep real secrets in `.env` (git-ignored).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    index_dir: Path = Path("./index")
    log_level: str = "info"


def get_settings() -> Settings:
    # Kept as a function so you can later add caching/lazy-loading if needed.
    return Settings()

