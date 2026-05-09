import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.services.subtitle import JOBS, process_video

router = APIRouter(prefix="/subtitle", tags=["subtitle"])
UPLOAD_DIR = Path("uploads")

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv"}


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    src_lang: str = Form("ko"),
    tgt_lang: str = Form("en"),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"지원하지 않는 파일 형식: {suffix}")

    UPLOAD_DIR.mkdir(exist_ok=True)
    job_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    async with aiofiles.open(video_path, "wb") as f:
        await f.write(await file.read())

    background_tasks.add_task(
        process_video,
        job_id,
        video_path,
        src_lang,
        tgt_lang,
        request.app.state.stt_service,
        request.app.state.translation_service,
    )

    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "존재하지 않는 작업입니다")
    job = JOBS[job_id]
    result = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done":
        result["download_url"] = f"/subtitle/download/{job_id}"
    if job["status"] == "error":
        result["error"] = job.get("error")
    return result


@router.get("/download/{job_id}")
async def download_srt(job_id: str):
    if job_id not in JOBS or JOBS[job_id]["status"] != "done":
        raise HTTPException(404, "아직 준비되지 않았습니다")
    return FileResponse(
        JOBS[job_id]["srt_path"],
        filename=f"{job_id}.srt",
        media_type="text/plain; charset=utf-8",
    )
