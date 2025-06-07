from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import dotenv_values


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str
    qdrant_url: str
    qdrant_key: str
    s3_bucket: str
    aws_key: str
    aws_secret: str
    redis_url: str


def _load_env(env_file: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Load environment variables using dotenv_values with optional env file."""
    path = Path(env_file) if env_file else None
    env = dotenv_values(path)
    return {**env, **os.environ}


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    """Return Settings instance with values sourced from environment variables."""
    env = _load_env(env_file)

    return Settings(
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        qdrant_url=env.get("QDRANT_URL", ""),
        qdrant_key=env.get("QDRANT_KEY", ""),
        s3_bucket=env.get("S3_BUCKET", ""),
        aws_key=env.get("AWS_KEY", ""),
        aws_secret=env.get("AWS_SECRET", ""),
        redis_url=env.get("REDIS_URL", ""),
    )


settings = load_settings()
