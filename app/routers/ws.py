import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])

MIN_TRANSLATE_CHARS = 6

# 연결된 학생 뷰어 목록
_viewers: set[WebSocket] = set()


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

    async def translate_and_send(text: str) -> bool:
        text = text.strip()
        if len(text) < MIN_TRANSLATE_CHARS:
            return False
        result = await translation_service.translate(text, src_lang, tgt_lang, glossary)
        if result.lower().startswith("please provide") or result.lower().startswith("i need the"):
            return False
        msg = {"type": "translation", "text": result}
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

            # STT 청크마다 바로 번역
            await translate_and_send(stt_text)

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
