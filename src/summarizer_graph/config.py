from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Конфигурация приложения."""
    openai_api_key: str
    summarizer_model: str = "gpt-4"
    predictor_model: str = "gpt-4"
    max_summary_tokens: int = 150
    max_prediction_tokens: int = 50
    temperature: float = 0

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
