"""Evaluate the active Transformers continuous-batching app against a source/reference set."""

import argparse
import asyncio
import csv
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin

import aiohttp


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_pairs(path: Path) -> list[dict[str, str]]:
    """Read source/reference pairs from CSV or JSON supplied by the evaluator."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("samples", payload.get("data", payload.get("test_cases")))
            if rows is None:
                raise ValueError("JSON object must contain samples, data, or test_cases")
        else:
            rows = payload
    else:
        raise ValueError("--data must be a .csv or .json file")

    source_keys = ("src", "source", "source_text", "input", "sentence_kor_Hang", "ko")
    reference_keys = ("ref", "reference", "reference_text", "target", "sentence_eng_Latn", "en")
    pairs = []
    for index, row in enumerate(rows, 1):
        source = next((row.get(key) for key in source_keys if row.get(key)), None)
        references = [row.get(key) for key in ("reference_a", "reference_b")
                      if isinstance(row.get(key), str) and row[key].strip()]
        if not references:
            reference = next((row.get(key) for key in reference_keys if row.get(key)), None)
            if isinstance(reference, str):
                references = [reference]
        if not isinstance(source, str) or not references:
            raise ValueError(
                f"row {index} needs source/src and reference/ref text columns"
            )
        pairs.append({"src": source.strip(), "refs": [ref.strip() for ref in references]})
    if not pairs:
        raise ValueError("evaluation data contains no source/reference pairs")
    return pairs


def parse_prometheus_metrics(raw: str) -> dict[tuple[str, str | None], float]:
    parsed = {}
    for line in raw.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        metric, value = line.rsplit(None, 1)
        if "{" in metric and metric.endswith("}"):
            name, label_text = metric[:-1].split("{", 1)
            le_match = re.search(r'\ble="([^"]+)"', label_text)
            le = le_match.group(1) if le_match else None
        else:
            name, le = metric, None
        try:
            parsed[(name, le)] = float(value)
        except ValueError:
            continue
    return parsed


def histogram_delta_summary(before: dict, after: dict, name: str) -> dict:
    buckets = []
    for (metric, le), current in after.items():
        if metric != f"{name}_bucket":
            continue
        previous = before.get((metric, le), 0.0)
        buckets.append((float(le) if le != "+Inf" else math.inf,
                        max(0.0, current - previous)))
    buckets.sort(key=lambda item: item[0])
    count = max(0.0, after.get((f"{name}_count", None), 0.0)
                - before.get((f"{name}_count", None), 0.0))
    total = max(0.0, after.get((f"{name}_sum", None), 0.0)
                - before.get((f"{name}_sum", None), 0.0))

    def quantile(q: float) -> float | None:
        if count <= 0 or not buckets:
            return None
        rank = count * q
        cumulative = 0.0
        previous_bound = 0.0
        previous_count = 0.0
        for bound, cumulative_count in buckets:
            if cumulative_count >= rank:
                in_bucket = cumulative_count - previous_count
                if math.isinf(bound) or in_bucket <= 0:
                    return previous_bound
                fraction = max(0.0, min(1.0, (rank - previous_count) / in_bucket))
                return previous_bound + (bound - previous_bound) * fraction
            previous_bound = bound
            previous_count = cumulative_count
            cumulative = cumulative_count
        return previous_bound if cumulative else None

    return {
        "count": int(count),
        "average_seconds": round(total / count, 4) if count else None,
        "p95_seconds_bucket_estimate": round(quantile(0.95), 4) if count else None,
        "p99_seconds_bucket_estimate": round(quantile(0.99), 4) if count else None,
    }


async def get_text(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def translate_one(session, endpoint, source, semaphore):
    async with semaphore:
        started = time.perf_counter()
        try:
            async with session.post(endpoint, json={
                "text": source,
                "src_lang": "ko",
                "tgt_lang": "en",
            }) as response:
                response.raise_for_status()
                payload = await response.json()
            return {
                "hypothesis": payload["translated"].strip(),
                "latency_seconds": time.perf_counter() - started,
                "error": None,
            }
        except Exception as exc:
            return {
                "hypothesis": "",
                "latency_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }


def calculate_comet(pairs: list[dict[str, str]], python_path: Path, timeout: int):
    helper = Path(__file__).with_name("score_comet.py")
    if not python_path.exists():
        return None, (
            f"COMET scorer not installed at {python_path}; create it with "
            "requirements-comet-eval.txt"
        )
    with tempfile.TemporaryDirectory(prefix="comet-eval-") as temp_dir:
        input_path = Path(temp_dir) / "pairs.json"
        output_path = Path(temp_dir) / "score.json"
        input_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
        try:
            result = subprocess.run(
                [str(python_path), str(helper), "--input", str(input_path),
                 "--output", str(output_path)],
                capture_output=True,
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if result.returncode != 0:
            return None, (result.stderr or result.stdout).strip()[-3000:]
        return json.loads(output_path.read_text(encoding="utf-8"))["COMET"], None


async def run_evaluation(args):
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as exc:
        raise RuntimeError(
            "평가 의존성이 없습니다. .venv/bin/pip install -r requirements-eval.txt 실행 필요"
        ) from exc

    data_path = Path(args.data)
    pairs = load_pairs(data_path)
    sources = [pair["src"] for pair in pairs]
    base_url = args.url.rstrip("/") + "/"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(urljoin(base_url, "health")) as response:
            response.raise_for_status()
            health = await response.json()
        before_text = await get_text(session, urljoin(base_url, "metrics"))

        endpoint = urljoin(base_url, "translate")
        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()
        tasks = [
            asyncio.create_task(translate_one(session, endpoint, source, semaphore))
            for source in sources
        ]
        completed = 0

        async def report_progress(task):
            nonlocal completed
            result = await task
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"번역 진행 {completed}/{len(tasks)}", flush=True)
            return result

        outputs = await asyncio.gather(*(report_progress(task) for task in tasks))
        total_time = time.perf_counter() - started
        after_text = await get_text(session, urljoin(base_url, "metrics"))

    hypotheses = [output["hypothesis"] for output in outputs]
    latencies = [output["latency_seconds"] for output in outputs]
    reference_streams = [
        [pair["refs"][min(index, len(pair["refs"]) - 1)] for pair in pairs]
        for index in range(max(len(pair["refs"]) for pair in pairs))
    ]
    bleu_score = BLEU(tokenize="13a").corpus_score(hypotheses, reference_streams).score
    chrf_score = CHRF().corpus_score(hypotheses, reference_streams).score
    comet_score = None
    comet_error = None
    comet_python = Path(args.comet_python)
    print("분리된 COMET 환경에서 CPU 점수화를 시도합니다...", flush=True)
    comet_score, comet_error = calculate_comet(
        [{"src": pair["src"], "mt": output["hypothesis"], "ref": ref}
         for pair, output in zip(pairs, outputs) for ref in pair["refs"]],
        comet_python,
        args.comet_timeout,
    )
    if comet_error:
        print(f"COMET 미실행: {comet_error}", flush=True)

    before_metrics = parse_prometheus_metrics(before_text)
    after_metrics = parse_prometheus_metrics(after_text)
    metrics = {
        "translation_batch_wait": histogram_delta_summary(
            before_metrics, after_metrics, "translation_batch_wait_seconds"
        ),
        "translation_generation": histogram_delta_summary(
            before_metrics, after_metrics, "translation_generation_seconds"
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_name(output_path.stem + "-metrics-before.txt").write_text(
        before_text, encoding="utf-8"
    )
    output_path.with_name(output_path.stem + "-metrics-after.txt").write_text(
        after_text, encoding="utf-8"
    )

    results = {
        "backend": health["backend"],
        "dataset_file": str(data_path),
        "model": "google/translategemma-4b-it",
        "source_language": "ko",
        "target_language": "en",
        "num_samples": len(pairs),
        "concurrency": args.concurrency,
        "request_error_count": sum(output["error"] is not None for output in outputs),
        "total_wall_time_seconds": round(total_time, 3),
        "throughput_sentences_per_second": round(len(pairs) / total_time, 4)
            if total_time else None,
        "latency_seconds": {
            "average": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 4) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 4) if latencies else None,
            "p99": round(percentile(latencies, 0.99), 4) if latencies else None,
        },
        "BLEU": round(bleu_score, 2),
        "chrF": round(chrf_score, 2),
        "comet_scorer_python": str(comet_python),
        "COMET": round(float(comet_score), 4) if comet_score is not None else None,
        "COMET_error": comet_error,
        "prometheus_histograms": metrics,
        "samples": [
            {
                "src": pair["src"],
                "hyp": output["hypothesis"],
                "refs": pair["refs"],
                "latency_seconds": round(output["latency_seconds"], 4),
                "error": output["error"],
            }
            for pair, output in zip(pairs, outputs)
        ],
    }
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n평가 완료")
    print(f"BLEU: {results['BLEU']} | chrF: {results['chrF']} | COMET: {results['COMET']}")
    print(
        "지연(s): 평균 {average}, p50 {p50}, p95 {p95}, p99 {p99}".format(
            **results["latency_seconds"]
        )
    )
    print(f"동시성: {args.concurrency} | 처리량: {results['throughput_sentences_per_second']} 문장/s")
    print(f"배치 대기 메트릭: {metrics['translation_batch_wait']}")
    print(f"결과 저장: {output_path}")
    latency_avg = results["latency_seconds"]["average"]
    comet_display = f"{results['COMET']:.4f}" if results["COMET"] is not None else "N/A"
    table = [
        "+----------------------------------+----------------------+-------+-------+--------+------------+------------------+",
        "| 시스템                           | 유형                 | BLEU  | chrF  | COMET  | 지연/문장  | 문장 처리량     |",
        "+----------------------------------+----------------------+-------+-------+--------+------------+------------------+",
        f"| TranslateGemma (우리 앱)         | 오프라인 · 배칭 x{args.concurrency:<2} | {results['BLEU']:>5.2f} | {results['chrF']:>5.2f} | {comet_display:>6} | {latency_avg:>8.2f}s | {results['throughput_sentences_per_second']:>8.3f} 문장/s |",
        "+----------------------------------+----------------------+-------+-------+--------+------------+------------------+",
    ]
    print("\n" + "\n".join(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--data", default="eval/test.json",
                        help="CSV or JSON with Korean source and English reference columns")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--comet-python", default=".venv-comet/bin/python",
                        help="Python executable in the isolated COMET environment")
    parser.add_argument("--comet-timeout", type=int, default=3600)
    parser.add_argument("--output", default="eval/results-transformers-cb.json")
    args = parser.parse_args()
    if args.concurrency < 1 or args.comet_timeout < 1:
        parser.error("--concurrency and --comet-timeout must be positive")
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
