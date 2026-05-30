"""
TranslateGemma 번역 품질 평가 스크립트
- 데이터셋: FLORES-200 (Korean → English)
- 지표: BLEU, chrF, COMET
- 대상: VLLM 서버 (localhost:8001)
"""

import json
import time
import requests
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoProcessor

VLLM_URL    = "http://localhost:8001/v1/completions"
MODEL_NAME  = "google/translategemma-4b-it"
NUM_SAMPLES = 200  # 전체 1012개 중 200개 테스트 (시간 절약)
OUTPUT_FILE = "eval_results.json"

_processor: AutoProcessor | None = None


def get_processor() -> AutoProcessor:
    global _processor
    if _processor is None:
        print("  AutoProcessor 로드 중...")
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return _processor


def translate(text: str) -> str:
    processor = get_processor()
    messages = [{"role": "user", "content": [
        {"type": "text", "source_lang_code": "ko",
         "target_lang_code": "en", "text": text}
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    try:
        res = requests.post(VLLM_URL, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"  번역 오류: {e}")
        return ""


def main():
    print("=" * 60)
    print("  TranslateGemma 4B 번역 품질 평가")
    print(f"  데이터셋: FLORES-200 Ko→En ({NUM_SAMPLES}개)")
    print("=" * 60)

    # FLORES-200 Ko-En 데이터셋 로드
    print("\n[1/4] 데이터셋 로딩 중...")
    dataset = load_dataset("facebook/flores", "kor_Hang-eng_Latn")
    devtest = dataset["devtest"]
    samples = list(devtest)[:NUM_SAMPLES]
    sources    = [s["sentence_kor_Hang"] for s in samples]
    references = [s["sentence_eng_Latn"] for s in samples]
    print(f"      {len(sources)}개 문장 로드 완료")

    # 번역
    print(f"\n[2/4] 번역 중... ({NUM_SAMPLES}개, 시간이 걸립니다)")
    hypotheses = []
    start = time.time()
    for i, src in enumerate(sources):
        hyp = translate(src)
        hypotheses.append(hyp)
        elapsed = time.time() - start
        avg_sec = elapsed / (i + 1)
        remain  = avg_sec * (NUM_SAMPLES - i - 1)
        print(f"  [{i+1:3d}/{NUM_SAMPLES}] ({remain/60:.1f}분 남음)  {src[:30]}...")
        print(f"           → {hyp[:60]}")

    total_time = time.time() - start
    print(f"\n  번역 완료: {total_time:.1f}초 ({total_time/NUM_SAMPLES:.2f}초/문장)")

    # BLEU, chrF 계산
    print("\n[3/4] BLEU / chrF 계산 중...")
    bleu = BLEU(tokenize="13a")
    chrf = CHRF()

    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])

    print(f"  BLEU : {bleu_score.score:.2f}")
    print(f"  chrF : {chrf_score.score:.2f}")

    # COMET 계산
    print("\n[4/4] COMET 계산 중... (모델 다운로드 필요할 수 있음)")
    comet_score = None
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
        comet_score  = comet_output.system_score
        print(f"  COMET: {comet_score:.4f}")
    except Exception as e:
        print(f"  COMET 스킵: {e}")

    # 결과 저장
    print(f"\n결과 저장 중... → {OUTPUT_FILE}")
    results = {
        "dataset":     "FLORES-200 Ko-En",
        "model":       MODEL_NAME,
        "num_samples": NUM_SAMPLES,
        "total_time_sec": round(total_time, 2),
        "sec_per_sentence": round(total_time / NUM_SAMPLES, 2),
        "BLEU":  round(bleu_score.score, 2),
        "chrF":  round(chrf_score.score, 2),
        "COMET": round(comet_score, 4) if comet_score else None,
        "samples": [
            {"src": s, "hyp": h, "ref": r}
            for s, h, r in zip(sources[:20], hypotheses[:20], references[:20])
        ]
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  최종 결과")
    print("=" * 60)
    print(f"  BLEU  : {results['BLEU']}")
    print(f"  chrF  : {results['chrF']}")
    print(f"  COMET : {results['COMET']}")
    print(f"  속도  : {results['sec_per_sentence']}초/문장")
    print(f"  결과 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
