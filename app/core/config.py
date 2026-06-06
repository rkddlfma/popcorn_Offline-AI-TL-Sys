from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 번역 모델
    translate_model: str = "google/translategemma-4b-it"
    translate_backend: str = "transformers"  # "transformers" | "vllm"

    # Whisper
    whisper_model: str = "base"

    # VLLM
    vllm_url: str = "http://localhost:8001"

    class Config:
        env_file = ".env"


settings = Settings()
