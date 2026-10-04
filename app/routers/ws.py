import asyncio
import json
import time
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.metrics import (
    audio_chunks,
    audio_ingress_queue_depth,
    audio_queue_backpressure,
    audio_queue_backpressure_events,
    audio_queue_wait,
    pipeline_errors,
    stage_latency,
    stt_completed_chunks,
    stt_nonempty_chunks,
    translation_completed_chunks,
    translation_eligible_chunks,
    translation_skipped_chunks,
    viewer_connections,
)

router = APIRouter(tags=["websocket"])

MIN_TRANSLATE_CHARS = 6
PARTIAL_FLUSH_CHARS = 6  # 부분 번역을 전송하는 누적 글자 증가 임계값 (스로틀)

# room_id별로 연결된 학생 뷰어 목록
_viewers_by_room: dict[str, set[WebSocket]] = {}


def _normalize_room_id(room_id: object) -> str:
    """1~5번 room ID만 허용합니다."""
    if not isinstance(room_id, str):
        raise ValueError("room_id는 문자열이어야 합니다")
    room_id = room_id.strip()
    if room_id not in {"1", "2", "3", "4", "5"}:
        raise ValueError("room_id는 1~5 중 하나여야 합니다")
    return room_id


async def _broadcast(room_id: str, message: dict) -> None:
    """해당 room의 뷰어에게만 병렬 전송합니다."""
    viewers = list(_viewers_by_room.get(room_id, ()))
    if not viewers:
        return
    payload = json.dumps(message)
    results = await asyncio.gather(
        *(v.send_text(payload) for v in viewers), return_exceptions=True
    )
    room_viewers = _viewers_by_room.get(room_id)
    if room_viewers is not None:
        room_viewers.difference_update(
            v for v, r in zip(viewers, results) if isinstance(r, Exception)
        )
        if not room_viewers:
            _viewers_by_room.pop(room_id, None)


@router.websocket("/ws/subtitle")
async def subtitle_ws(websocket: WebSocket):
    """교수용 WebSocket — 마이크 오디오 수신 → STT → 번역 → 브로드캐스트."""
    await websocket.accept()
    stt_service         = websocket.app.state.stt_service
    translation_service = websocket.app.state.translation_service

    config   = json.loads(await websocket.receive_text())
    try:
        room_id = _normalize_room_id(config.get("room_id", "1"))
    except ValueError:
        await websocket.close(code=1008, reason="유효하지 않은 room_id")
        return
    src_lang = config.get("src_lang", "ko")
    tgt_lang = config.get("tgt_lang", "en")
    glossary = config.get("glossary") or None  # {"원문": "번역"} | None
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    audio_ingress_queue_depth.labels(room_id).set(0)
    backend = getattr(translation_service, "_backend", None)
    continuous_batching_active = bool(
        getattr(backend, "continuous_batching_active", False)
    )
    translation_parallelism = asyncio.Semaphore(2)
    translation_condition = asyncio.Condition()
    translation_results: dict[int, dict] = {}
    translation_sequence = {"submitted": 0, "delivered": 0, "stopping": False}
    translation_tasks: set[asyncio.Task] = set()

    def _is_hallucination(text: str) -> bool:
        low = text.lower()
        return low.startswith("please provide") or low.startswith("i need the")

    async def _send(msg: dict) -> None:
        started = time.perf_counter()
        await websocket.send_text(json.dumps(msg))
        await _broadcast(room_id, msg)
        if msg.get("type") == "translation" and msg.get("done"):
            stage_latency.labels("translation_to_student_send").observe(time.perf_counter() - started)

    async def translate_and_send(
        text: str, event_id: str, timing: dict, queued_mono: float
    ) -> bool:
        text = text.strip()
        if len(text) < MIN_TRANSLATE_CHARS:
            translation_skipped_chunks.labels(room_id, "short_text").inc()
            return False
        translation_eligible_chunks.labels(room_id).inc()

        acc = ""
        decided = False        # 헛소리 prefix 판정 완료 여부
        last_sent_len = 0      # 마지막으로 전송한 누적 길이 (스로틀용)
        # 토큰 스트리밍: 부분 번역을 즉시 흘려보내 체감 지연을 줄임
        async for acc in translation_service.translate_stream(
            text, src_lang, tgt_lang, glossary, priority="high"
        ):
            if not decided:
                if _is_hallucination(acc):
                    translation_skipped_chunks.labels(room_id, "hallucination").inc()
                    return False
                if len(acc) < MIN_TRANSLATE_CHARS:
                    continue   # 판정 보류 — 아직 전송하지 않음
                decided = True
            if len(acc) - last_sent_len >= PARTIAL_FLUSH_CHARS:
                last_sent_len = len(acc)
                message_timing = {**timing, "translation_streamed_at": time.time(),
                                  "student_delivery_started_at": time.time()}
                await _send({"type": "translation", "text": acc, "done": False,
                             "event_id": event_id, "timing": message_timing})

        acc = acc.strip()
        if not acc or _is_hallucination(acc):
            translation_skipped_chunks.labels(room_id, "invalid_output").inc()
            return False
        translated_at = time.time()
        delivery_started_at = time.time()
        stage_latency.labels("queue_to_translation_complete").observe(time.perf_counter() - queued_mono)
        await _send({"type": "translation", "text": acc, "done": True,
                     "event_id": event_id,
                     "timing": {**timing, "translation_completed_at": translated_at,
                                "student_delivery_started_at": delivery_started_at}})
        translation_completed_chunks.labels(room_id).inc()
        return True

    async def collect_translation(
        sequence: int, text: str, event_id: str, timing: dict, queued_mono: float
    ) -> None:
        outcome = {
            "status": "skipped", "partials": [], "text": "", "error": "",
            "event_id": event_id, "timing": timing,
        }
        acc = ""
        decided = False
        last_sent_len = 0
        hallucination_rejected = False
        try:
            async for acc in translation_service.translate_stream(
                text, src_lang, tgt_lang, glossary, priority="high"
            ):
                if not decided:
                    if _is_hallucination(acc):
                        translation_skipped_chunks.labels(room_id, "hallucination").inc()
                        hallucination_rejected = True
                        break
                    if len(acc) < MIN_TRANSLATE_CHARS:
                        continue
                    decided = True
                if len(acc) - last_sent_len >= PARTIAL_FLUSH_CHARS:
                    last_sent_len = len(acc)
                    outcome["partials"].append(acc)

            acc = acc.strip()
            if acc and not hallucination_rejected and not _is_hallucination(acc):
                outcome["status"] = "complete"
                outcome["text"] = acc
                stage_latency.labels("queue_to_translation_complete").observe(
                    time.perf_counter() - queued_mono
                )
            elif not hallucination_rejected:
                reason = "empty_output" if not acc else "hallucination"
                translation_skipped_chunks.labels(room_id, reason).inc()
        except Exception as exc:
            pipeline_errors.labels("translation").inc()
            outcome["status"] = "error"
            outcome["error"] = str(exc)
        finally:
            async with translation_condition:
                translation_results[sequence] = outcome
                translation_condition.notify_all()

    async def deliver_ordered_translations() -> None:
        while True:
            async with translation_condition:
                await translation_condition.wait_for(
                    lambda: (
                        translation_sequence["delivered"]
                        in translation_results
                    ) or (
                        translation_sequence["stopping"]
                        and translation_sequence["delivered"]
                        >= translation_sequence["submitted"]
                    )
                )
                if (translation_sequence["stopping"]
                        and translation_sequence["delivered"]
                        >= translation_sequence["submitted"]):
                    return
                sequence = translation_sequence["delivered"]
                outcome = translation_results.pop(sequence)
                translation_sequence["delivered"] += 1

            try:
                if outcome["status"] == "error":
                    await websocket.send_text(json.dumps({
                        "type": "error", "text": outcome["error"]
                    }))
                elif outcome["status"] == "complete":
                    for partial in outcome["partials"]:
                        message_timing = {
                            **outcome["timing"],
                            "translation_streamed_at": time.time(),
                            "student_delivery_started_at": time.time(),
                        }
                        await _send({
                            "type": "translation", "text": partial, "done": False,
                            "event_id": outcome["event_id"], "timing": message_timing,
                        })
                    translated_at = time.time()
                    delivery_started_at = time.time()
                    await _send({
                        "type": "translation", "text": outcome["text"], "done": True,
                        "event_id": outcome["event_id"],
                        "timing": {
                            **outcome["timing"],
                            "translation_completed_at": translated_at,
                            "student_delivery_started_at": delivery_started_at,
                        },
                    })
                    translation_completed_chunks.labels(room_id).inc()
            except Exception:
                pipeline_errors.labels("websocket_delivery").inc()
            finally:
                translation_parallelism.release()

    async def enqueue_translation(
        text: str, event_id: str, timing: dict, queued_mono: float
    ) -> None:
        await translation_parallelism.acquire()
        sequence = translation_sequence["submitted"]
        translation_sequence["submitted"] += 1
        translation_eligible_chunks.labels(room_id).inc()
        task = asyncio.create_task(collect_translation(
            sequence, text, event_id, timing, queued_mono
        ))
        translation_tasks.add(task)
        task.add_done_callback(translation_tasks.discard)
        async with translation_condition:
            translation_condition.notify_all()

    async def process_audio_chunk(item: tuple) -> None:
        data, audio_received_at, audio_received_mono, event_id, enqueued_mono = item
        audio = np.frombuffer(data, dtype=np.float32)

        stt_started = time.perf_counter()
        try:
            stt_text = await stt_service.transcribe(audio, src_lang)
        except Exception:
            pipeline_errors.labels("stt").inc()
            raise
        stt_completed_at = time.time()
        stt_completed_mono = time.perf_counter()
        stt_completed_chunks.labels(room_id).inc()
        stage_latency.labels("audio_to_stt").observe(stt_completed_mono - audio_received_mono)
        if not stt_text.strip():
            translation_skipped_chunks.labels(room_id, "empty_stt").inc()
            return
        stt_nonempty_chunks.labels(room_id).inc()

        await _send({"type": "stt", "text": stt_text, "event_id": event_id,
                     "timing": {"audio_received_at": audio_received_at,
                                "stt_completed_at": stt_completed_at,
                                "stt_duration_ms": round((stt_completed_mono-stt_started)*1000, 2)}})

        translation_queued_at = time.time()
        queued_mono = time.perf_counter()
        stage_latency.labels("stt_to_translation_queue").observe(queued_mono - stt_completed_mono)
        timing = {
            "audio_received_at": audio_received_at,
            "stt_completed_at": stt_completed_at,
            "translation_queued_at": translation_queued_at,
        }
        if continuous_batching_active and len(stt_text.strip()) >= MIN_TRANSLATE_CHARS:
            await enqueue_translation(stt_text, event_id, timing, queued_mono)
            return
        try:
            await translate_and_send(stt_text, event_id, timing, queued_mono)
        except Exception:
            pipeline_errors.labels("translation").inc()
            raise

    async def receive_audio() -> None:
        while True:
            try:
                data  = await websocket.receive_bytes()
            except WebSocketDisconnect:
                return
            except Exception as e:
                try:
                    await websocket.send_text(json.dumps({"type": "error", "text": str(e)}))
                except Exception:
                    return
                continue

            audio_received_at = time.time()
            audio_received_mono = time.perf_counter()
            enqueued_mono = time.perf_counter()
            event_id = uuid.uuid4().hex
            audio_chunks.labels(room_id).inc()
            if audio_queue.full():
                audio_queue_backpressure_events.labels(room_id).inc()
            put_started = time.perf_counter()
            await audio_queue.put((
                data, audio_received_at, audio_received_mono, event_id, enqueued_mono
            ))
            audio_queue_backpressure.labels(room_id).observe(
                time.perf_counter() - put_started
            )
            audio_ingress_queue_depth.labels(room_id).set(audio_queue.qsize())

    async def process_audio_queue() -> None:
        while True:
            item = await audio_queue.get()
            audio_ingress_queue_depth.labels(room_id).set(audio_queue.qsize())
            audio_queue_wait.labels(room_id).observe(time.perf_counter() - item[4])
            try:
                await process_audio_chunk(item)
            except Exception as e:
                try:
                    await websocket.send_text(json.dumps({"type": "error", "text": str(e)}))
                except Exception:
                    pass
            finally:
                audio_queue.task_done()
                audio_ingress_queue_depth.labels(room_id).set(audio_queue.qsize())

    worker = asyncio.create_task(process_audio_queue())
    dispatcher = (
        asyncio.create_task(deliver_ordered_translations())
        if continuous_batching_active else None
    )
    try:
        await receive_audio()
        await audio_queue.join()
        if continuous_batching_active:
            while translation_tasks:
                await asyncio.gather(*tuple(translation_tasks), return_exceptions=True)
            async with translation_condition:
                translation_sequence["stopping"] = True
                translation_condition.notify_all()
            await dispatcher
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        if dispatcher is not None and not dispatcher.done():
            dispatcher.cancel()
            await asyncio.gather(dispatcher, return_exceptions=True)
        audio_ingress_queue_depth.labels(room_id).set(0)


@router.websocket("/ws/viewer")
async def viewer_ws(websocket: WebSocket):
    """학생용 WebSocket — 번역 결과 수신만."""
    try:
        room_id = _normalize_room_id(websocket.query_params.get("room_id", "1"))
    except ValueError:
        await websocket.close(code=1008, reason="유효하지 않은 room_id")
        return

    await websocket.accept()
    _viewers_by_room.setdefault(room_id, set()).add(websocket)
    viewer_connections.labels(room_id).inc()
    try:
        while True:
            # 연결 유지용 (클라이언트가 끊으면 예외 발생)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        room_viewers = _viewers_by_room.get(room_id)
        if room_viewers is not None:
            room_viewers.discard(websocket)
            if not room_viewers:
                _viewers_by_room.pop(room_id, None)
        viewer_connections.labels(room_id).dec()
