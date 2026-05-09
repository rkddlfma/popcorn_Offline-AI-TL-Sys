import asyncio

import numpy as np
import torch
from faster_whisper import WhisperModel

from app.core.config import settings


class STTService:
    def __init__(self):
        self._model: WhisperModel | None = None

    def load(self) -> None:
        print(f"[STTService] Whisper 로드 중: {settings.whisper_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            settings.whisper_model, device=device, compute_type=compute
        )
        print("[STTService] 로드 완료")

    async def transcribe(self, audio: np.ndarray, language: str) -> str:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe_sync, audio, language
        )

    def _transcribe_sync(self, audio: np.ndarray, language: str) -> str:
        segments, _ = self._model.transcribe(audio, language=language, beam_size=5, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments)

    def transcribe_segments(self, audio_path: str, language: str) -> list[dict]:
        """타임스탬프 포함 세그먼트 반환 (배치 자막용)."""
        segments, _ = self._model.transcribe(audio_path, language=language, beam_size=5, vad_filter=True)
        return [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments]
