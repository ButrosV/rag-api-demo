from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """
    RAG API configuration settings.

    Loads from environment variables and .env file (see .env.example).
    Required secrets go in .env.
    """

    model_config = SettingsConfigDict(env_file=str(DOTENV_PATH), extra="ignore", case_sensitive=False)

    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    index_dir: Path = Path("./index")
    log_level: str = "info"


def get_settings() -> Settings:
    """
    Factory function to create Settings instance.
    Loads configuration from .env file and environment variables with hardcoded defaults.
    :return: Settings instance for API and ingestion use.
    """
    return Settings()
