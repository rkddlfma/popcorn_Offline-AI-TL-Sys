import json
import re
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings

router = APIRouter(tags=["websocket"])

SENTENCE_END = re.compile(r'(?<=[.!?,，。！？])\s*')
MIN_TRANSLATE_CHARS = 6


def _split_sentences(text: str) -> tuple[list[str], str]:
    parts = SENTENCE_END.split(text.strip())
    complete = [p for p in parts[:-1] if p.strip()]
    remainder = parts[-1].strip() if parts else ""
    if remainder and remainder[-1] in set('.!?,，。！？'):
        complete.append(remainder)
        remainder = ""
    return complete, remainder


@router.websocket("/ws/subtitle")
async def subtitle_ws(websocket: WebSocket):
    """실시간 자막 WebSocket 엔드포인트.

    프로토콜:
      1) 연결 직후 클라이언트가 JSON 설정 전송 (1회)
         {"src_lang": "ko", "tgt_lang": "en"}
      2) 이후 오디오를 float32 바이너리 프레임으로 전송

    서버 → 클라이언트:
      {"type": "stt",         "text": "..."}
      {"type": "translation", "text": "..."}
      {"type": "error",       "text": "..."}
    """
    await websocket.accept()
    stt_service         = websocket.app.state.stt_service
    translation_service = websocket.app.state.translation_service

    config   = json.loads(await websocket.receive_text())
    src_lang = config.get("src_lang", "ko")
    tgt_lang = config.get("tgt_lang", "en")

    text_buffer:  str         = ""
    buffer_start: float | None = None

    async def translate_and_send(text: str) -> bool:
        text = text.strip()
        if len(text) < MIN_TRANSLATE_CHARS:
            return False
        result = await translation_service.translate(text, src_lang, tgt_lang)
        if result.lower().startswith("please provide") or result.lower().startswith("i need the"):
            return False
        await websocket.send_text(json.dumps({"type": "translation", "text": result}))
        return True

    try:
        while True:
            data  = await websocket.receive_bytes()
            audio = np.frombuffer(data, dtype=np.float32)

            stt_text = await stt_service.transcribe(audio, src_lang)
            if not stt_text.strip():
                continue

            await websocket.send_text(json.dumps({"type": "stt", "text": stt_text}))

            text_buffer = (text_buffer + " " + stt_text).strip()
            if buffer_start is None:
                buffer_start = time.time()

            sentences, remainder = _split_sentences(text_buffer)

            for sentence in sentences:
                ok = await translate_and_send(sentence)
                if not ok:
                    remainder = (sentence + " " + remainder).strip()

            text_buffer = remainder

            if text_buffer and buffer_start and (time.time() - buffer_start) >= settings.max_buffer_sec:
                ok = await translate_and_send(text_buffer)
                if ok:
                    text_buffer  = ""
                    buffer_start = None
            elif not text_buffer:
                buffer_start = None

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "text": str(e)}))
        except Exception:
            pass
