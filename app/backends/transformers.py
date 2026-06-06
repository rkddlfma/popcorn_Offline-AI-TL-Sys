import asyncio
import threading
from typing import AsyncIterator

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from app.backends.base import TranslationBackend
from app.core.config import settings

# bitsandbytes 버전 호환 패치
try:
    from bitsandbytes.nn import Params4bit as _P4b
    _orig_new = _P4b.__new__
    def _patched_new(cls, *args, _is_hf_initialized=None, **kwargs):
        return _orig_new(cls, *args, **kwargs)
    _P4b.__new__ = _patched_new
except Exception:
    pass


class TransformersBackend(TranslationBackend):
    def __init__(self):
        self._processor = None
        self._model = None
        self._gpu_lock = asyncio.Lock()  # GPU는 1개 — 번역 직렬화
        self._realtime_waiting = 0       # 대기 중인 실시간(high) 요청 수

    def load(self) -> None:
        print(f"[TransformersBackend] 모델 로드 중: {settings.translate_model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._processor = AutoProcessor.from_pretrained(settings.translate_model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            settings.translate_model,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        self._model.eval()
        print("[TransformersBackend] 로드 완료")

    # ── 우선순위 게이트 ────────────────────────────────────────────────────
    async def _acquire_priority(self, priority: str) -> None:
        """배치(normal) 요청은 대기 중인 실시간(high) 요청에 GPU를 양보."""
        if priority == "high":
            self._realtime_waiting += 1
        else:
            while self._realtime_waiting > 0:
                await asyncio.sleep(0.05)

    def _release_priority(self, priority: str) -> None:
        if priority == "high":
            self._realtime_waiting -= 1

    # ── 번역 (전체 결과 반환) ──────────────────────────────────────────────
    async def translate(
        self, text: str, src_lang: str, tgt_lang: str, priority: str = "normal"
    ) -> str:
        await self._acquire_priority(priority)
        try:
            async with self._gpu_lock:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._translate_sync, text, src_lang, tgt_lang
                )
        finally:
            self._release_priority(priority)

    # ── 번역 (토큰 스트리밍, 누적 문자열 yield) ────────────────────────────
    async def translate_stream(
        self, text: str, src_lang: str, tgt_lang: str, priority: str = "normal"
    ) -> AsyncIterator[str]:
        await self._acquire_priority(priority)
        try:
            async with self._gpu_lock:
                loop = asyncio.get_event_loop()
                inputs = await loop.run_in_executor(
                    None, self._build_inputs, text, src_lang, tgt_lang
                )
                streamer = TextIteratorStreamer(
                    self._processor, skip_special_tokens=True, skip_prompt=True
                )
                gen_kwargs = dict(
                    **inputs, streamer=streamer, do_sample=False, max_new_tokens=256
                )
                thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
                thread.start()

                sentinel = object()

                def _next():
                    try:
                        return next(streamer)
                    except StopIteration:
                        return sentinel

                acc = ""
                while True:
                    token = await loop.run_in_executor(None, _next)
                    if token is sentinel:
                        break
                    acc += token
                    yield acc.strip()
                thread.join()
        finally:
            self._release_priority(priority)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────
    def _build_inputs(self, text: str, src_lang: str, tgt_lang: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_lang,
                        "target_lang_code": tgt_lang,
                        "text": text,
                    }
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

    def _translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
        inputs = self._build_inputs(text, src_lang, tgt_lang)
        streamer = TextIteratorStreamer(
            self._processor, skip_special_tokens=True, skip_prompt=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            do_sample=False,
            max_new_tokens=256,
        )
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        result = ""
        for token in streamer:
            result += token
        thread.join()

        return result.strip()
