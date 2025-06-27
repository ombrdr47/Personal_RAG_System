# config/settings.py   ← keep the filename the same

import os
from pathlib import Path                # ← NEW
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Google API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: Path = Path("./data/vector_db")

    # 📂 **NEW** — where to save uploaded files
    DATA_DIR: Path = Path("./data")      # override with env var if you like:  DATA_DIR=/mnt/persist

    # Document Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.7

    # Memory
    CONVERSATION_MEMORY_SIZE: int = 10

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
