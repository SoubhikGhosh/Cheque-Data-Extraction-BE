from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_ID: str
    LOCATION: str
    API_ENDPOINT: str
    LOG_LEVEL: str = "INFO"
    MAX_WORKERS: int = 80
    BATCH_SIZE: int = 40

    # Safety settings (can be moved to constants if preferred, but good for config if they change)
    # For simplicity, keeping the structure from the original code.
    # A more Pydantic way would be a nested model.
    HARM_CATEGORY_HATE_SPEECH_THRESHOLD: str = "BLOCK_NONE" # OFF maps to BLOCK_NONE
    HARM_CATEGORY_DANGEROUS_CONTENT_THRESHOLD: str = "BLOCK_NONE"
    HARM_CATEGORY_SEXUALLY_EXPLICIT_THRESHOLD: str = "BLOCK_NONE"
    HARM_CATEGORY_HARASSMENT_THRESHOLD: str = "BLOCK_NONE"


    CORS_ALLOW_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    APP_TITLE: str = "Cheque Data Extraction API"
    APP_DESCRIPTION: str = "API for processing zip files containing cheque images using Vertex AI"
    APP_VERSION: str = "2.0.0"


    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()