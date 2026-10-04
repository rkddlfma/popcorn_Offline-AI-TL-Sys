"""Low-cardinality Prometheus metrics for realtime speech translation."""
from prometheus_client import Counter, Gauge, Histogram

_ROOMS = ("1", "2", "3", "4", "5")
_LATENCY = (0.01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60)

viewer_connections = Gauge("subtitle_viewer_connections", "Connected viewer sockets", ["room_id"])
for _room in _ROOMS:
    viewer_connections.labels(_room).set(0)
audio_chunks = Counter("subtitle_audio_chunks_total", "Audio chunks received", ["room_id"])
stage_latency = Histogram("subtitle_stage_latency_seconds", "Realtime pipeline stage duration", ["stage"], buckets=_LATENCY)
translation_queue_wait = Histogram("translation_gpu_lock_wait_seconds", "Time waiting to acquire translation GPU lock", buckets=_LATENCY)
translation_inference = Histogram("translation_inference_seconds", "Time from translation lock acquisition to stream completion", buckets=_LATENCY)
translation_active = Gauge("translation_active_requests", "Requests inside translation lock")
stt_active = Gauge("stt_active_requests", "STT requests currently executing")
gpu_overlap = Gauge("gpu_stt_translation_overlap", "1 when STT and translation execute concurrently")
pipeline_errors = Counter("subtitle_pipeline_errors_total", "Realtime pipeline errors", ["stage"])

_stt_count = 0
_translation_count = 0


def stt_enter() -> None:
    global _stt_count
    _stt_count += 1
    stt_active.set(_stt_count)
    _set_overlap()


def stt_exit() -> None:
    global _stt_count
    _stt_count = max(0, _stt_count - 1)
    stt_active.set(_stt_count)
    _set_overlap()


def translation_enter() -> None:
    global _translation_count
    _translation_count += 1
    translation_active.set(_translation_count)
    _set_overlap()


def translation_exit() -> None:
    global _translation_count
    _translation_count = max(0, _translation_count - 1)
    translation_active.set(_translation_count)
    _set_overlap()


def _set_overlap() -> None:
    gpu_overlap.set(1 if _stt_count and _translation_count else 0)
