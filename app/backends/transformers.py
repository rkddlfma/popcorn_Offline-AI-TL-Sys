import asyncio
import copy
import threading
import time
import uuid
from typing import AsyncIterator

import torch
from app.metrics import (
    translation_batch_wait,
    translation_enter,
    translation_exit,
    translation_generation,
    translation_inference,
    translation_queue_wait,
)
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
    def __init__(self, continuous_batching: bool = True):
        self._processor = None
        self._model = None
        self._continuous_batching = continuous_batching
        self._continuous_manager = None
        self._gpu_lock = asyncio.Lock()  # GPU는 1개 — 번역 직렬화
        self._realtime_waiting = 0       # 대기 중인 실시간(high) 요청 수

    @property
    def continuous_batching_active(self) -> bool:
        return self._continuous_manager is not None

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
        if self._continuous_batching:
            try:
                from transformers import ContinuousBatchingConfig
            except ImportError as exc:
                raise RuntimeError(
                    "Continuous batching requires a newer Transformers release"
                ) from exc
            if not hasattr(self._model, "init_continuous_batching"):
                raise RuntimeError(
                    "Loaded Transformers model does not support continuous batching"
                )
            generation_config = copy.deepcopy(self._model.generation_config)
            generation_config.do_sample = False
            generation_config.max_new_tokens = 256
            self._continuous_manager = self._model.init_continuous_batching(
                generation_config=generation_config,
                continuous_batching_config=ContinuousBatchingConfig(
                    max_requests_per_batch=4,
                    max_queue_size=128,
                    max_memory_percent=0.5,
                ),
            )
            self._continuous_manager.warmup()
            self._continuous_manager.start()
        print("[TransformersBackend] 로드 완료")

    def close(self) -> None:
        if self._continuous_manager is not None:
            self._continuous_manager.stop(block=True, timeout=30)
            self._continuous_manager.destroy()
            self._continuous_manager = None

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
        if self._continuous_manager is not None:
            result = ""
            async for partial in self.translate_stream(
                text, src_lang, tgt_lang, priority
            ):
                result = partial
            return result

        await self._acquire_priority(priority)
        try:
            queued_at = asyncio.get_running_loop().time()
            async with self._gpu_lock:
                translation_queue_wait.observe(asyncio.get_running_loop().time() - queued_at)
                translation_enter()
                started = asyncio.get_running_loop().time()
                try:
                    return await asyncio.get_event_loop().run_in_executor(
                        None, self._translate_sync, text, src_lang, tgt_lang
                    )
                finally:
                    translation_inference.observe(asyncio.get_running_loop().time() - started)
                    translation_exit()
        finally:
            self._release_priority(priority)

    # ── 번역 (토큰 스트리밍, 누적 문자열 yield) ────────────────────────────
    async def translate_stream(
        self, text: str, src_lang: str, tgt_lang: str, priority: str = "normal"
    ) -> AsyncIterator[str]:
        await self._acquire_priority(priority)
        try:
            if self._continuous_manager is not None:
                async for partial in self._translate_stream_continuous(
                    text, src_lang, tgt_lang
                ):
                    yield partial
                return

            queued_at = asyncio.get_running_loop().time()
            async with self._gpu_lock:
                translation_queue_wait.observe(asyncio.get_running_loop().time() - queued_at)
                translation_enter()
                started = asyncio.get_running_loop().time()
                loop = asyncio.get_event_loop()
                try:
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
                    translation_inference.observe(asyncio.get_running_loop().time() - started)
                    translation_exit()
        finally:
            self._release_priority(priority)

    async def _translate_stream_continuous(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> AsyncIterator[str]:
        """Submit a request to the shared Transformers continuous-batching manager."""
        loop = asyncio.get_running_loop()
        inputs = await loop.run_in_executor(
            None, self._build_inputs, text, src_lang, tgt_lang
        )
        input_ids = inputs["input_ids"][0].tolist()
        request_id = uuid.uuid4().hex
        outputs: asyncio.Queue = asyncio.Queue()
        manager = self._continuous_manager
        submitted_at = time.perf_counter()
        first_output = True
        generation_started_at = None

        def on_output(result) -> None:
            outputs.put_nowait(result)

        translation_enter()
        completed = False
        try:
            manager.register_result_handler(request_id, on_output)
            accepted = await loop.run_in_executor(
                None,
                lambda: manager.add_request(
                    input_ids=input_ids,
                    request_id=request_id,
                    max_new_tokens=256,
                    streaming=True,
                ),
            )
            if accepted is None:
                raise RuntimeError("Continuous batching scheduler rejected request")

            while True:
                result = await outputs.get()
                if result.error:
                    raise RuntimeError(result.error)
                if first_output and result.lifespan[0] > 0:
                    generation_started_at = result.lifespan[0]
                    translation_batch_wait.observe(
                        max(0.0, result.lifespan[0] - submitted_at)
                    )
                    first_output = False

                partial = self._processor.decode(
                    result.generated_tokens, skip_special_tokens=True
                ).strip()
                if partial:
                    yield partial
                if result.is_finished():
                    completed = True
                    if generation_started_at is not None:
                        translation_generation.observe(
                            max(0.0, time.perf_counter() - generation_started_at)
                        )
                    break
        finally:
            if not completed and manager.is_running():
                manager.cancel_request(request_id)
            translation_exit()

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
        # 비스트리밍 경로 — 스트리머/스레드 없이 직접 생성 후 새 토큰만 디코딩
        inputs = self._build_inputs(text, src_lang, tgt_lang)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, do_sample=False, max_new_tokens=256
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
