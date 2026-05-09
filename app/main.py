from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from app.core.config import settings
from app.core.model_manager import create_translation_backend
from app.routers import translate, ws
from app.services.stt import STTService
from app.services.translation import TranslationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델을 한 번만 로드합니다."""
    print("=" * 50)
    print("  TranslateGemma API 서버 시작")
    print(f"  번역 백엔드 : {settings.translate_backend}")
    print(f"  Whisper     : {settings.whisper_model}")
    print(f"  CUDA        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print("=" * 50)

    # STT 모델 로드
    stt_service = STTService()
    stt_service.load()
    app.state.stt_service = stt_service

    # 번역 백엔드 + 서비스 로드
    backend = create_translation_backend()
    backend.load()
    translation_service = TranslationService(backend)
    app.state.translation_service = translation_service

    print("\n서버 준비 완료.\n")
    yield
    # 종료 시 정리 작업 (필요 시 추가)


app = FastAPI(
    title="TranslateGemma API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(translate.router)
app.include_router(ws.router)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.translate_backend}
