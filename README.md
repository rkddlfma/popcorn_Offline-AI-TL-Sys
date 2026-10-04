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
| `transformers` | Transformers continuous batching으로 4-bit 모델 실행 | Gemma 3 지원 및 최신 Transformers 필요, batch 상한 4 |
| `transformers_legacy` | 단일 요청 GPU Lock 경로 | 이전 동작 비교나 롤백용 |
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

번역 품질/지연 평가기는 `eval/test.json`의 원문-정답 쌍을 현재 실행 중인 FastAPI 앱에
보냅니다. `.env`에서 `TRANSLATE_BACKEND=transformers`로 앱을 실행하세요. VLLM 서버는
필요하지 않습니다. COMET의 Transformers 4 의존성이 앱의 Transformers 5와 충돌하지
않도록 별도 가상환경을 사용합니다.

```bash
.venv/bin/pip install -r requirements-eval.txt
python3 -m venv .venv-comet
.venv-comet/bin/pip install -r requirements-comet-eval.txt
.venv/bin/python evaluate.py --url http://localhost:8000 --data eval/test.json --concurrency 3
```

JSON은 `[ {"src": "한국어 원문", "ref": "English reference"} ]` 형식이며,
`source`/`reference` 필드 이름도 지원합니다. CSV 입력이면 `src,ref` 열을 사용합니다.
평가기는 요청 3개를 동시에 보내 배칭을 유도하고, BLEU/chrF/COMET,
요청 지연 평균/p50/p95/p99 및 Prometheus batch 대기/생성 histogram 변화를 JSON으로
저장합니다. COMET은 Transformers 버전 충돌을 막기 위해 `.venv-comet`에서 CPU로
계산합니다. COMET 모델 가중치 최초 다운로드에는 인터넷 연결이 필요합니다.

최근 결과는 [eval_results.json](eval_results.json), [eval_e2e_results.json](eval_e2e_results.json) 참고.

## 동시성 부하 측정

Room별 WebSocket 및 GPU 병목 측정 방법은 [부하 측정 가이드](docs/load-testing.md)를 참고하세요. WAV 기반 부하 스크립트는 `scripts/load_test.py`이며 Prometheus 메트릭은 `/metrics`에서 확인할 수 있습니다.
