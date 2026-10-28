import sys
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure backend directory is in sys.path so 'from src...' imports work seamlessly from any CWD
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load .env: prioritize backend/.env, fallback to root .env
if (BASE_DIR / ".env").exists():
    load_dotenv(BASE_DIR / ".env")
elif (BASE_DIR.parent / ".env").exists():
    load_dotenv(BASE_DIR.parent / ".env")
else:
    load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable bindings."""

    # API Keys (Free-tier providers)
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")

    # Primary Model Routing (Groq active production models: groq/openai/gpt-oss-120b, groq/mistral-saba-24b)
    generator_model: str = Field(
        default="groq/openai/gpt-oss-120b",
        alias="GENERATOR_MODEL",
        description="Default candidate generator model via LiteLLM",
    )
    judge_model: str = Field(
        default="gemini/gemini-1.5-flash",
        alias="JUDGE_MODEL",
        description="Default cultural alignment judge model via LiteLLM",
    )

    # Embedding & Vector Database
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    chroma_persist_dir: str = Field(
        default=str(BASE_DIR / "vector_store" / "chroma_db"),
        alias="CHROMA_PERSIST_DIR",
    )

    # Cultural Alignment Rules & Thresholds
    score_threshold: float = Field(
        default=0.75,
        alias="SCORE_THRESHOLD",
        description="Minimum overall score to accept candidate response",
    )
    max_feedback_loops: int = Field(
        default=2,
        alias="MAX_FEEDBACK_LOOPS",
        description="Maximum regeneration attempts per user query",
    )

    # Data paths
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"

    model_config = SettingsConfigDict(
        env_file=[str(BASE_DIR / ".env"), str(BASE_DIR.parent / ".env"), ".env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
