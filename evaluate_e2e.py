"""
STT → 번역 엔드-투-엔드 평가 스크립트
- 데이터셋: FLORES-200 kor_Hang-eng_Latn 한국어 문장을 gTTS로 음성 합성
           → Whisper STT → TranslateGemma 번역 → FLORES-200 영어 정답 비교
- STT    : faster-whisper (base, 백엔드와 동일)
- 번역   : TranslateGemma (VLLM /v1/completions)
- 지표   : WER (STT), BLEU / chrF / COMET (번역), 지연시간

사전 준비:
  pip install jiwer gtts
  sudo apt install ffmpeg   (faster-whisper MP3 디코딩에 필요)
"""

import json
import os
import tempfile
import time

import requests
import torch
from datasets import load_dataset
from faster_whisper import WhisperModel
from gtts import gTTS
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoProcessor

VLLM_URL      = "http://localhost:8001/v1/completions"
MODEL_NAME    = "google/translategemma-4b-it"
WHISPER_MODEL = "base"
NUM_SAMPLES   = 100
OUTPUT_FILE   = "eval_e2e_results.json"


# ── 모델 초기화 ───────────────────────────────────────────────────────────────

def load_models():
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"

    print(f"[초기화] faster-whisper {WHISPER_MODEL} 로드 중 (device: {device})...")
    whisper = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)

    print("[초기화] AutoProcessor 로드 중...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("[초기화] 완료\n")
    return whisper, processor


# ── TTS ──────────────────────────────────────────────────────────────────────

def text_to_audio_file(text: str) -> str:
    """gTTS로 한국어 텍스트 → 임시 MP3 파일 경로 반환."""
    tts = gTTS(text=text, lang="ko")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    return tmp.name


# ── STT ──────────────────────────────────────────────────────────────────────

def transcribe(whisper: WhisperModel, audio_path: str) -> str:
    """faster-whisper로 한국어 STT (파일 경로 직접 전달)."""
    segments, _ = whisper.transcribe(
        audio_path, language="ko", beam_size=5, vad_filter=True
    )
    return " ".join(seg.text.strip() for seg in segments)


# ── 번역 ─────────────────────────────────────────────────────────────────────

def translate(processor: AutoProcessor, text: str) -> str:
    if not text.strip():
        return ""
    messages = [{"role": "user", "content": [
        {"type": "text", "source_lang_code": "ko",
         "target_lang_code": "en", "text": text}
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    payload = {"model": MODEL_NAME, "prompt": prompt,
               "max_tokens": 512, "temperature": 0.0}
    try:
        res = requests.post(VLLM_URL, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"  번역 오류: {e}")
        return ""


# ── WER ──────────────────────────────────────────────────────────────────────

def compute_wer(hyps: list[str], refs: list[str]) -> float | None:
    try:
        import jiwer
        transform = jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        return jiwer.wer(refs, hyps,
                         reference_transform=transform,
                         hypothesis_transform=transform) * 100
    except ImportError:
        print("  WER 스킵 — pip install jiwer")
        return None


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STT → 번역 엔드-투-엔드 평가")
    print(f"  데이터셋 : FLORES-200 Ko→En (gTTS 음성합성, {NUM_SAMPLES}개)")
    print(f"  STT      : faster-whisper-{WHISPER_MODEL}")
    print(f"  번역     : {MODEL_NAME}")
    print("=" * 60)

    whisper, processor = load_models()

    # FLORES-200 로드 (한국어 문장 + 영어 정답)
    print("[1/5] 데이터셋 로딩 중...")
    dataset = load_dataset("facebook/flores", "kor_Hang-eng_Latn", split="devtest")
    samples = list(dataset)[:NUM_SAMPLES]
    print(f"  {len(samples)}개 샘플 로드 완료\n")

    stt_hyps   = []
    stt_refs   = []
    trans_hyps = []
    trans_refs = []
    latencies  = []
    saved      = []

    print("[2/5] TTS 음성합성 → STT → 번역 파이프라인 실행 중...")
    for i, sample in enumerate(samples):
        src_ko    = sample["sentence_kor_Hang"]   # 한국어 원문
        trans_ref = sample["sentence_eng_Latn"]   # 영어 정답

        t0 = time.time()

        # TTS: 한국어 텍스트 → MP3 파일
        audio_path = text_to_audio_file(src_ko)
        try:
            # STT: MP3 → 한국어 텍스트
            stt_hyp = transcribe(whisper, audio_path)
            # 번역: 한국어 텍스트 → 영어
            trans_hyp = translate(processor, stt_hyp)
        finally:
            os.unlink(audio_path)

        elapsed = time.time() - t0
        latencies.append(elapsed)

        stt_hyps.append(stt_hyp)
        stt_refs.append(src_ko)       # WER: STT 결과 vs 원문 한국어
        trans_hyps.append(trans_hyp)
        trans_refs.append(trans_ref)

        avg_lat = sum(latencies) / len(latencies)
        remain  = avg_lat * (NUM_SAMPLES - i - 1)
        print(f"  [{i+1:3d}/{NUM_SAMPLES}] {elapsed:.2f}s  ({remain/60:.1f}분 남음)")
        print(f"    STT  : {stt_hyp[:55]}")
        print(f"    번역 : {trans_hyp[:55]}")

        if i < 20:
            saved.append({
                "src_ko":    src_ko,
                "stt_hyp":   stt_hyp,
                "trans_ref": trans_ref,
                "trans_hyp": trans_hyp,
                "latency_s": round(elapsed, 3),
            })

    # WER
    print("\n[3/5] WER 계산 중...")
    wer = compute_wer(stt_hyps, stt_refs)
    if wer is not None:
        print(f"  WER: {wer:.2f}%")

    # BLEU / chrF
    print("\n[4/5] BLEU / chrF 계산 중...")
    bleu_score = BLEU(tokenize="13a").corpus_score(trans_hyps, [trans_refs])
    chrf_score = CHRF().corpus_score(trans_hyps, [trans_refs])
    print(f"  BLEU : {bleu_score.score:.2f}")
    print(f"  chrF : {chrf_score.score:.2f}")

    # COMET
    print("\n[5/5] COMET 계산 중...")
    comet_score = None
    try:
        from comet import download_model, load_from_checkpoint
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        comet_data  = [{"src": s, "mt": h, "ref": r}
                       for s, h, r in zip(stt_refs, trans_hyps, trans_refs)]
        comet_score = comet_model.predict(comet_data, batch_size=8, gpus=1).system_score
        print(f"  COMET: {comet_score:.4f}")
    except Exception as e:
        print(f"  COMET 스킵: {e}")

    # 결과 저장
    results = {
        "dataset":       "FLORES-200 Ko→En (gTTS 음성합성)",
        "stt_model":     f"faster-whisper-{WHISPER_MODEL}",
        "trans_model":   MODEL_NAME,
        "num_samples":   NUM_SAMPLES,
        "avg_latency_s": round(sum(latencies) / len(latencies), 3),
        "WER":           round(wer, 2) if wer is not None else None,
        "BLEU":          round(bleu_score.score, 2),
        "chrF":          round(chrf_score.score, 2),
        "COMET":         round(comet_score, 4) if comet_score else None,
        "note":          "텍스트 전용 BLEU 21.44 대비 STT 오류 누적 영향 측정",
        "samples":       saved,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  최종 결과")
    print("=" * 60)
    print(f"  WER      : {results['WER']}%  ← STT 정확도 (낮을수록 좋음)")
    print(f"  BLEU     : {results['BLEU']}  (텍스트 전용: 21.44)")
    print(f"  chrF     : {results['chrF']}  (텍스트 전용: 53.96)")
    print(f"  COMET    : {results['COMET']}  (텍스트 전용: 0.8692)")
    print(f"  지연시간  : {results['avg_latency_s']}초/샘플  (TTS+STT+번역)")
    print(f"  결과 파일 : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
