import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])

MIN_TRANSLATE_CHARS = 6
PARTIAL_FLUSH_CHARS = 6  # 부분 번역을 전송하는 누적 글자 증가 임계값 (스로틀)

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

    def _is_hallucination(text: str) -> bool:
        low = text.lower()
        return low.startswith("please provide") or low.startswith("i need the")

    async def _send(msg: dict) -> None:
        await websocket.send_text(json.dumps(msg))
        await _broadcast(msg)

    async def translate_and_send(text: str) -> bool:
        text = text.strip()
        if len(text) < MIN_TRANSLATE_CHARS:
            return False

        acc = ""
        decided = False        # 헛소리 prefix 판정 완료 여부
        last_sent_len = 0      # 마지막으로 전송한 누적 길이 (스로틀용)
        # 토큰 스트리밍: 부분 번역을 즉시 흘려보내 체감 지연을 줄임
        async for acc in translation_service.translate_stream(
            text, src_lang, tgt_lang, glossary, priority="high"
        ):
            if not decided:
                if _is_hallucination(acc):
                    return False
                if len(acc) < MIN_TRANSLATE_CHARS:
                    continue   # 판정 보류 — 아직 전송하지 않음
                decided = True
            if len(acc) - last_sent_len >= PARTIAL_FLUSH_CHARS:
                last_sent_len = len(acc)
                await _send({"type": "translation", "text": acc, "done": False})

        acc = acc.strip()
        if not acc or _is_hallucination(acc):
            return False
        await _send({"type": "translation", "text": acc, "done": True})
        return True

    try:
        while True:
            try:
                data  = await websocket.receive_bytes()
                audio = np.frombuffer(data, dtype=np.float32)

                stt_text = await stt_service.transcribe(audio, src_lang)
                if not stt_text.strip():
                    continue

                await _send({"type": "stt", "text": stt_text})

                # STT 청크마다 바로 번역 (스트리밍)
                await translate_and_send(stt_text)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                # 개별 청크 처리 오류는 세션을 끊지 않고 해당 청크만 건너뜀
                try:
                    await websocket.send_text(json.dumps({"type": "error", "text": str(e)}))
                except Exception:
                    break
    except WebSocketDisconnect:
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
