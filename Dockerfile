# CUDA 런타임 베이스 — bitsandbytes 4-bit / GPU 추론에 필요
# (RTX 50xx 블랙웰은 CUDA 12.8+ / torch 2.7+ 가 필요할 수 있음 — 그 경우 베이스 태그와
#  requirements 의 torch 버전을 함께 올릴 것)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Python + ffmpeg 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY app/ ./app/
COPY static/ ./static/

# 업로드/자막 저장 디렉토리
RUN mkdir -p uploads subtitles

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
