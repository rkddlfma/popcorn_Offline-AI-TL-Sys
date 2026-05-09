import asyncio
from pathlib import Path

JOBS: dict = {}  # job_id → {status, srt_path, error}
SUBTITLES_DIR = Path("subtitles")


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

        # ffmpeg로 16kHz mono WAV 추출
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(video_path),
            "-ar", "16000", "-ac", "1", "-y", str(audio_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg 오디오 추출 실패")

        JOBS[job_id]["status"] = "transcribing"
        segments = await asyncio.get_event_loop().run_in_executor(
            None, stt_service.transcribe_segments, str(audio_path), src_lang
        )

        JOBS[job_id]["status"] = "translating"
        srt_entries = []
        for i, seg in enumerate(segments, 1):
            if not seg["text"]:
                continue
            translated = await translation_service.translate(seg["text"], src_lang, tgt_lang)
            srt_entries.append(
                f"{i}\n{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n{translated}\n"
            )

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


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
