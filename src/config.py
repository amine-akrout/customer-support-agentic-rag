"""Configuration settings for the project."""

import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Project settings."""

    model_config = SettingsConfigDict(
        env_file="./.env", env_file_encoding="utf-8", extra="allow"
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "indexes"

    # Data settings
    DATA_URL: str = (
        "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    RAW_DATA_PATH: str = str(DATA_DIR / "customer_care_emails.csv")
    PROCESSED_DATA_PATH: str = str(DATA_DIR / "processed_data.csv")

    # Kaggle settings

    # Embeddings settings
    EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM settings
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    LLM_MAX_TOKENS: int = 100

    # Local LLM settings
    OLLAMA_MODEL_NAME: str = "llama3.2:1b"

    FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss_index.faiss")

    FAISS_TOP_K: int = 5

    # Open AI API settings
    OPENAI_API_KEY: Union[SecretStr, None] = None

    # Logging settings
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FILE: str = str(BASE_DIR / "logs" / "preprocessing.log")

    # Ensure that the data directory exists
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.INDEX_DIR, exist_ok=True)
        os.makedirs(self.BASE_DIR / "logs", exist_ok=True)


settings = Settings()
