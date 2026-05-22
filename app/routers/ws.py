import json
import re
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings

router = APIRouter(tags=["websocket"])

SENTENCE_END = re.compile(r'(?<=[.!?,，。！？])\s*')
MIN_TRANSLATE_CHARS = 6

# 연결된 학생 뷰어 목록
_viewers: set[WebSocket] = set()


def _split_sentences(text: str) -> tuple[list[str], str]:
    parts = SENTENCE_END.split(text.strip())
    complete = [p for p in parts[:-1] if p.strip()]
    remainder = parts[-1].strip() if parts else ""
    if remainder and remainder[-1] in set('.!?,，。！？'):
        complete.append(remainder)
        remainder = ""
    return complete, remainder


async def _broadcast(message: dict) -> None:
    """연결된 모든 뷰어에게 메시지 전송."""
    disconnected = set()
    for viewer in _viewers:
        try:
            await viewer.send_text(json.dumps(message))
        except Exception:
            disconnected.add(viewer)
    _viewers.difference_update(disconnected)


@router.websocket("/ws/subtitle")
async def subtitle_ws(websocket: WebSocket):
    """교수용 WebSocket — 마이크 오디오 수신 → STT → 번역 → 브로드캐스트."""
    await websocket.accept()
    stt_service         = websocket.app.state.stt_service
    translation_service = websocket.app.state.translation_service

    config   = json.loads(await websocket.receive_text())
    src_lang = config.get("src_lang", "ko")
    tgt_lang = config.get("tgt_lang", "en")
    glossary = config.get("glossary") or None  # {"원문": "번역"} | None

    text_buffer:  str         = ""
    buffer_start: float | None = None

    async def translate_and_send(text: str) -> bool:
        text = text.strip()
        if len(text) < MIN_TRANSLATE_CHARS:
            return False
        result = await translation_service.translate(text, src_lang, tgt_lang, glossary)
        if result.lower().startswith("please provide") or result.lower().startswith("i need the"):
            return False
        msg = {"type": "translation", "text": result}
        # 교수 화면 + 모든 학생에게 동시 전송
        await websocket.send_text(json.dumps(msg))
        await _broadcast(msg)
        return True

    try:
        while True:
            data  = await websocket.receive_bytes()
            audio = np.frombuffer(data, dtype=np.float32)

            stt_text = await stt_service.transcribe(audio, src_lang)
            if not stt_text.strip():
                continue

            stt_msg = {"type": "stt", "text": stt_text}
            await websocket.send_text(json.dumps(stt_msg))
            await _broadcast(stt_msg)

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


@router.websocket("/ws/viewer")
async def viewer_ws(websocket: WebSocket):
    """학생용 WebSocket — 번역 결과 수신만."""
    await websocket.accept()
    _viewers.add(websocket)
    try:
        while True:
            # 연결 유지용 (클라이언트가 끊으면 예외 발생)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _viewers.discard(websocket)
