import asyncio
import threading

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
        self._lock = asyncio.Lock()  # 동시 번역 요청 직렬화

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

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._translate_sync, text, src_lang, tgt_lang
            )

    def _translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
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
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

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
