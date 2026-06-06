import json
from typing import AsyncIterator

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
        self._session: aiohttp.ClientSession | None = None

    def load(self) -> None:
        print(f"[VLLMBackend] 프로세서 로드 중: {settings.translate_model}")
        # 프롬프트 포맷팅용으로만 로드 (모델 가중치는 VLLM 서버가 보유)
        self._processor = AutoProcessor.from_pretrained(settings.translate_model)
        print(f"[VLLMBackend] VLLM 서버: {self._url}")
        print("[VLLMBackend] 로드 완료")

    async def _get_session(self) -> aiohttp.ClientSession:
        # 이벤트 루프가 떠 있는 첫 요청 시점에 세션을 만들어 재사용 (커넥션 풀 유지)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _build_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": src_lang,
             "target_lang_code": tgt_lang, "text": text}
        ]}]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    async def translate(
        self, text: str, src_lang: str, tgt_lang: str, priority: str = "normal"
    ) -> str:
        payload = {
            "model": settings.translate_model,
            "prompt": self._build_prompt(text, src_lang, tgt_lang),
            "max_tokens": 256,
            "temperature": 0,
        }

        session = await self._get_session()
        async with session.post(
            f"{self._url}/v1/completions", json=payload
        ) as resp:
            data = await resp.json()
            if "choices" not in data:
                raise RuntimeError(f"VLLM 응답 오류 (HTTP {resp.status}): {data}")
            return data["choices"][0]["text"].strip()

    async def translate_stream(
        self, text: str, src_lang: str, tgt_lang: str, priority: str = "normal"
    ) -> AsyncIterator[str]:
        """OpenAI 호환 completions SSE(stream=True)로 토큰을 받아 누적 문자열을 yield."""
        payload = {
            "model": settings.translate_model,
            "prompt": self._build_prompt(text, src_lang, tgt_lang),
            "max_tokens": 256,
            "temperature": 0,
            "stream": True,
        }

        session = await self._get_session()
        acc = ""
        buffer = b""
        async with session.post(
            f"{self._url}/v1/completions", json=payload
        ) as resp:
            if resp.status != 200:
                detail = (await resp.read()).decode(errors="ignore")
                raise RuntimeError(f"VLLM 스트리밍 오류 (HTTP {resp.status}): {detail}")

            # SSE: "data: {json}\n\n" 형식. 청크가 줄 경계로 안 떨어지므로 직접 버퍼링.
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices")
                    if not choices:
                        continue
                    token = choices[0].get("text", "")
                    if token:
                        acc += token
                        yield acc.strip()
