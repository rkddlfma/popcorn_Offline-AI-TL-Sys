# 동시성 및 GPU 병목 부하 측정

## 계측 항목

서버의 `GET /metrics`는 Prometheus text exposition 형식으로 제공합니다.

| Metric | 의미 |
|---|---|
| `subtitle_viewer_connections{room_id}` | Room별 현재 WebSocket 학생 연결 수 |
| `subtitle_audio_chunks_total{room_id}` | 수신한 오디오 청크 수 |
| `subtitle_stage_latency_seconds{stage}` | 오디오→STT, STT→번역 큐, 큐→번역 완료, 번역 완료→학생 소켓 송신까지의 histogram |
| `translation_gpu_lock_wait_seconds` | 우선순위 게이트 이후 Transformers GPU Lock 획득까지 대기한 시간 |
| `translation_inference_seconds` | Lock 보유 시간(입력 전처리 + 생성 + 스트림 소비 포함) |
| `stt_active_requests`, `translation_active_requests` | 동시에 진행 중인 각 작업 수 |
| `gpu_stt_translation_overlap` | STT 작업과 번역 Lock 점유가 겹치는 동안 1 |
| `subtitle_pipeline_errors_total{stage}` | STT 또는 WebSocket 청크 처리 오류 |

Histogram의 `_bucket`, `_sum`, `_count` 시계열로 평균과 p50/p95/p99를 계산합니다. Prometheus histogram은 서버 측 percentile을 직접 내지 않으므로 scrape 기간별 `histogram_quantile()`을 사용하세요. 예:

```promql
histogram_quantile(0.95, sum by (le) (rate(translation_gpu_lock_wait_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(translation_inference_seconds_bucket[5m])))
sum by (room_id) (subtitle_viewer_connections)
```

서버는 오디오 수신, STT 완료, 번역 큐 진입, 번역 완료, 학생 송신 시작의 epoch timestamp를 WebSocket 메시지의 `timing`에 담습니다. `/metrics`의 `translation_to_student_send`는 서버가 교수 소켓과 Room 학생 소켓에 쓰기를 완료하는 데 걸린 시간입니다. 부하 스크립트는 서버와 클라이언트가 같은 호스트 시계를 쓸 때 수신 지연도 기록합니다. 별도 장비로 실행할 때는 각 브라우저 수신시각 및 해당 구간의 Prometheus 데이터를 저장하고 시계 동기화를 확인하세요. `gpu_stt_translation_overlap`은 STT executor 작업과 번역 Lock 점유의 동시성을 나타내며, 두 모델의 CUDA kernel 실행이 정확히 같은 순간 겹쳤다는 profiler 증거는 아닙니다. Nsight Systems/torch profiler를 병행하면 실제 GPU 실행 경합을 확인할 수 있습니다.

## 부하 실행

필요 오디오는 mono, 16 kHz, signed 16-bit PCM WAV 한국어 강의 녹음입니다. 모델을 올린 서버에서 별도 머신 또는 같은 호스트의 부하 클라이언트로 실행합니다.

```bash
pip install -r requirements.txt
python scripts/load_test.py --url http://localhost:8000 --audio lecture.wav
```

Room마다 다른 WAV를 재생하려면 `--room-audio ROOM=FILE` 옵션을 반복합니다. `--audio`는 개별 지정이 없는 room의 기본 파일이며 생략할 수 있습니다. 예를 들어:

```bash
python scripts/load_test.py --rooms 3 --room-audio 1=lecture-a.wav --room-audio 2=lecture-b.wav --room-audio 3=lecture-c.wav
```

`--audio default.wav`와 `--room-audio 2=lecture-b.wav`를 함께 쓰면 Room 2에는 `lecture-b.wav`, 나머지에는 `default.wav`가 재생됩니다. `--audio` 없이 실행할 때는 측정에 포함되는 모든 room에 파일을 지정해야 합니다.

### 다국어 Room 동시 부하

`scripts/multilingual_load_test.py`는 동일한 WAV를 Room 1~3에 동시에 전송하며, Room별 목표 언어를 영어(`en`), 일본어(`ja`), 중국어(`zh`)로 설정합니다. 기본값은 Room당 100명(총 300명), WAV는 `wav/brian-neutral-and-measured.wav`, 원문 언어는 영어입니다.

```bash
.venv/bin/python scripts/multilingual_load_test.py \
  --url http://localhost:8000 \
  --audio wav/brian-neutral-and-measured.wav \
  --src-lang en \
  --students-per-room 100 \
  --output multilingual-load-test-results.json
```

WAV 음성이 한국어라면 `--src-lang ko`로 바꾸세요. WAV는 mono, 16 kHz, signed 16-bit PCM이어야 합니다. 결과 JSON에는 연결된 뷰어 수와 전체/Room별 E2E 및 서버 단계별 지연 통계가 기록됩니다.

기본 실행은 1 Room (50명), 3 Rooms (150명), 5 Rooms (250명) 시나리오를 차례로 수행합니다. 각 교수 소켓은 지정된 WAV를 청크 단위 실시간 속도로 한 번 재생합니다. 재생을 반복하거나 청크 크기를 바꾸려면 `--chunk-seconds`를 조정하고 더 긴 WAV를 사용하세요. 개별 시나리오는 `--rooms 5`로 고를 수 있습니다. 기본 drain은 30초이며 가장 긴 번역 대기보다 길게 설정하세요. JSON 리포트에는 각 학생이 받은 완료 번역에 대해 오디오 수신부터 화면 수신까지의 클라이언트 지연과 서버 단계별 지연의 평균/p50/p95/p99가 기록됩니다. 클라이언트와 서버 시계가 같은 호스트 시계가 아니면 e2e 값은 시계 오차를 포함하므로, 각 브라우저에서 수신시각 및 해당 측정 구간의 Prometheus 데이터를 함께 저장합니다.

권장 절차:

1. 재시작 직후 모델 warm-up을 완료하고 `/metrics` counter/histogram 시작값을 저장합니다.
2. 1 Room 단계 실행 후 지표를 저장합니다. 3 Rooms, 5 Rooms도 같은 오디오, 청크 크기, 모델/백엔드, 하드웨어 조건으로 실행합니다.
3. 스크립트는 각 단계 후 기본 5초간 대기합니다(`--cooldown`). 각 단계는 최소 5분 또는 동일한 오디오 분량이 충분히 반복되도록 운용하고, GPU 사용률/메모리와 온도가 안정화됐는지 확인합니다. 기본 스크립트는 WAV 1회 재생이므로 반복 부하가 필요하면 길이가 긴 WAV로 진행합니다.
4. 각 단계의 `translation_gpu_lock_wait_seconds` 및 `translation_inference_seconds` histogram을 해당 scrape 구간으로 나눠 percentile을 계산하고, Room별 접속자가 목표치(50명)에 도달했는지 확인합니다.
5. `gpu_stt_translation_overlap`이 1이 되는 구간과 p95/p99 상승을 대조합니다. overlap은 동시 작업 관측 지표이므로 GPU profiler 결과와 함께 해석합니다.

## 배치 추론 도입 판단 기준

아래는 실시간 강의 자막의 초기 SLO 제안입니다. 실제 수업 네트워크 및 자막 허용 지연과 비교해 확정하세요. 세 수치를 모두 p95/p99로 판정하며, 동일 조건에서 3회 측정해 2회 이상 초과하면 병목으로 봅니다.

| 5 Room / 250명 기준 | 목표 | 조치 |
|---|---:|---|
| 오디오 수신→STT 완료 | p95 ≤ 2.0초, p99 ≤ 4.0초 | STT 청크 길이/VAD/Whisper 모델 또는 GPU 용량 점검 |
| GPU Lock 대기 | p95 ≤ 0.5초, p99 ≤ 1.0초 | 초과 시 번역 직렬화가 병목. Continuous batching 후보 |
| 번역 Lock 보유(입력+추론+스트리밍) | p95 ≤ 2.0초, p99 ≤ 4.0초 | 추론 자체가 지배적이면 batching/서빙 엔진 후보 |
| 오디오 수신→학생 완료 자막 수신 | p95 ≤ 4.0초, p99 ≤ 6.0초 | 초과 시 사용자 체감 SLO 실패; stage histogram으로 원인 분리 |
| 5 Rooms Lock 대기 증가율 | 1 Room 대비 p95 3배 초과 또는 절대 목표 초과 | GPU Lock 직렬화 비용이 유의함; Continuous Batching 실험 진행 |

Lock 대기만 높고 추론 시간이 낮다면 큐잉을 줄이는 배치/서빙 스케줄링의 효과를 우선 검증합니다. 추론 시간이 높다면 batching으로 처리량이 좋아져도 첫 토큰/문장 지연이 커질 수 있으므로, 5 Room 동일 부하에서 p95/p99 E2E가 개선되고 1 Room 지연이 회귀하지 않을 때 도입합니다. 현재 구현은 Transformers backend Lock 구간을 계측합니다. VLLM backend는 외부 서버 큐/추론을 사용하므로 해당 두 histogram은 0건이며, VLLM 서버의 자체 Prometheus 지표를 같은 구간에 수집해야 비교가 공정합니다.
