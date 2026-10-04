from app.backends.base import TranslationBackend
from app.backends.transformers import TransformersBackend
from app.backends.vllm import VLLMBackend
from app.core.config import settings


def create_translation_backend() -> TranslationBackend:
    if settings.translate_backend == "vllm":
        return VLLMBackend()
    if settings.translate_backend == "transformers_legacy":
        return TransformersBackend(continuous_batching=False)
    return TransformersBackend(continuous_batching=True)
