from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 번역 모델
    translate_model: str = "google/translategemma-4b-it"
    translate_backend: str = "transformers"  # "transformers" | "vllm"

    # Whisper
    whisper_model: str = "base"

    # VLLM
    vllm_url: str = "http://localhost:8001"

    # VAD
    vad_threshold: float = 0.02
    silence_duration: float = 0.7
    max_buffer_chars: int = 150
    max_buffer_sec: float = 6.0

    class Config:
        env_file = ".env"


settings = Settings()
