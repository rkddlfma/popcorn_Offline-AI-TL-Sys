# TranslateGemma — 오프라인 AI 번역 자막 시스템

강의 현장에서 사용하는 **완전 로컬(오프라인)** 번역 시스템입니다. 외부 AI API 없이
[TranslateGemma](https://huggingface.co/google/translategemma-4b-it)(4-bit 양자화)로 동작합니다.

## 주요 기능

1. **실시간 수업 자막** — 마이크 → Whisper STT → 번역 → WebSocket → 브라우저 자막 오버레이 (토큰 스트리밍)
2. **동영상 자동 자막** — 영상 업로드 → STT + 타임코드 정렬 → SRT 생성
3. **채팅/공지 번역 API** — 짧은 문장 번역 + 용어집(Glossary) 주입

## 기술 스택

- API: FastAPI + WebSocket
- STT: faster-whisper
- 번역: TranslateGemma (Transformers 4-bit **또는** VLLM 백엔드 — 교체 가능)
- 배포: Docker (GPU 패스스루)

## 빠른 시작 (로컬)

```bash
pip install -r requirements.txt
cp .env.example .env          # 백엔드/모델 설정
uvicorn app.main:app --reload
```

- 교수용 UI: http://localhost:8000/
- 학생용 자막 뷰어: http://localhost:8000/view
- API 문서: http://localhost:8000/docs

## 백엔드 선택

`.env` 의 `TRANSLATE_BACKEND` 로 전환합니다.

| 값 | 설명 | 비고 |
|----|------|------|
| `transformers` | 단일 프로세스에서 4-bit 모델 직접 로드 | 별도 서버 불필요 |
| `vllm` | 외부 VLLM 서버에 추론 위임 | 처리량 우선, 먼저 `vllm serve` 필요 |

VLLM 사용 시:

```bash
vllm serve google/translategemma-4b-it --port 8001
```

## Docker 실행 (GPU)

`nvidia-container-toolkit` 가 설치되어 있어야 합니다.

```bash
docker compose up --build
```

> RTX 50xx(블랙웰) GPU는 CUDA 12.8+ / torch 2.7+ 가 필요할 수 있습니다.
> 이 경우 `Dockerfile` 의 베이스 이미지 태그와 `requirements.txt` 의 torch 버전을 함께 올리세요.

## REST API 예시

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"머신러닝 강의를 시작합니다","src_lang":"ko","tgt_lang":"en"}'
```

## 평가

번역 품질/지연을 측정하는 스크립트가 포함되어 있습니다 (FLORES-200 기반).

```bash
python evaluate.py        # 텍스트 전용 BLEU/chrF/COMET
python evaluate_e2e.py    # TTS→STT→번역 엔드투엔드 (WER 포함)
```

최근 결과는 [eval_results.json](eval_results.json), [eval_e2e_results.json](eval_e2e_results.json) 참고.
