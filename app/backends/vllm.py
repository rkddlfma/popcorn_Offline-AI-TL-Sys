import aiohttp
from transformers import AutoProcessor

from app.backends.base import TranslationBackend
from app.core.config import settings


class VLLMBackend(TranslationBackend):
    """VLLM 백엔드.

    실행 전 VLLM 서버를 먼저 띄워야 합니다:
      vllm serve google/translategemma-4b-it --port 8001
    """

    def __init__(self):
        self._url = settings.vllm_url.rstrip("/")
        self._processor: AutoProcessor | None = None

    def load(self) -> None:
        print(f"[VLLMBackend] 프로세서 로드 중: {settings.translate_model}")
        # 프롬프트 포맷팅용으로만 로드 (모델 가중치는 VLLM 서버가 보유)
        self._processor = AutoProcessor.from_pretrained(settings.translate_model)
        print(f"[VLLMBackend] VLLM 서버: {self._url}")
        print("[VLLMBackend] 로드 완료")

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": src_lang,
             "target_lang_code": tgt_lang, "text": text}
        ]}]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        payload = {
            "model": settings.translate_model,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._url}/v1/completions", json=payload
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    raise RuntimeError(f"VLLM 응답 오류 (HTTP {resp.status}): {data}")
                return data["choices"][0]["text"].strip()
