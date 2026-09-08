import asyncio
import subprocess
from pathlib import Path

JOBS: dict = {}  # job_id → {status, srt_path, error}
SUBTITLES_DIR = Path("subtitles")
MAX_JOBS = 50  # 보관할 최대 작업 수 (초과 시 완료/오류 작업부터 정리)
MAX_CONCURRENT_TRANSLATIONS = 8  # 배치 번역 동시 요청 상한


def _prune_jobs() -> None:
    """완료/오류 상태의 오래된 작업을 제거해 메모리 누수를 막습니다."""
    if len(JOBS) <= MAX_JOBS:
        return
    removable = [jid for jid, j in JOBS.items() if j.get("status") in ("done", "error")]
    for jid in removable[: len(JOBS) - MAX_JOBS]:
        job = JOBS.pop(jid, None)
        if job and job.get("srt_path"):
            Path(job["srt_path"]).unlink(missing_ok=True)


async def process_video(
    job_id: str,
    video_path: Path,
    src_lang: str,
    tgt_lang: str,
    stt_service,
    translation_service,
) -> None:
    audio_path = video_path.with_suffix(".wav")
    try:
        JOBS[job_id] = {"status": "extracting"}

        # ffmpeg로 16kHz mono WAV 추출 (subprocess.run → thread executor로 안전하게 실행)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", "-y", str(audio_path)],
                capture_output=True,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg 오디오 추출 실패: " + result.stderr.decode(errors="ignore"))

        JOBS[job_id]["status"] = "transcribing"
        segments = await loop.run_in_executor(
            None, stt_service.transcribe_segments, str(audio_path), src_lang
        )

        JOBS[job_id]["status"] = "translating"
        valid_segments = [seg for seg in segments if seg["text"]]

        # 병렬 번역 — vLLM 백엔드는 continuous batching으로 동시 처리,
        # transformers 백엔드는 GPU 락이 직렬화하므로 동작은 동일하되 손해 없음.
        # 동시 요청 수를 제한해 vLLM 서버 과부하를 방지.
        sem = asyncio.Semaphore(MAX_CONCURRENT_TRANSLATIONS)

        async def _translate_one(seg: dict) -> str:
            async with sem:
                return await translation_service.translate(seg["text"], src_lang, tgt_lang)

        translations = await asyncio.gather(
            *(_translate_one(seg) for seg in valid_segments)
        )

        srt_entries = [
            f"{idx}\n{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n{translated}\n"
            for idx, (seg, translated) in enumerate(zip(valid_segments, translations), 1)
        ]

        SUBTITLES_DIR.mkdir(exist_ok=True)
        srt_path = SUBTITLES_DIR / f"{job_id}.srt"
        srt_path.write_text("\n".join(srt_entries), encoding="utf-8")

        JOBS[job_id] = {"status": "done", "srt_path": str(srt_path)}

    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}
    finally:
        video_path.unlink(missing_ok=True)
        if audio_path.exists():
            audio_path.unlink()
        _prune_jobs()


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
