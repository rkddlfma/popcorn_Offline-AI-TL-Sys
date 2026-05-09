from app.backends.base import TranslationBackend
from app.backends.transformers import TransformersBackend
from app.backends.vllm import VLLMBackend
from app.core.config import settings


def create_translation_backend() -> TranslationBackend:
    if settings.translate_backend == "vllm":
        return VLLMBackend()
    return TransformersBackend()
