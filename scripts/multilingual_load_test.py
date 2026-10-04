#!/usr/bin/env python3
"""Send one WAV concurrently to Rooms 1-3 with distinct target languages."""
import argparse
import asyncio
import json
import re
import time
import urllib.request

import websockets
from websockets.exceptions import ConnectionClosed

from load_test import read_audio, summary


LATENCY_NAMES = (
    "e2e_ms",
    "audio_to_stt",
    "stt_to_queue",
    "queue_to_delivery_server",
)
DEFAULT_TARGETS = {"1": "en", "2": "ja", "3": "zh"}


def read_metrics(url):
    with urllib.request.urlopen(f"{url.rstrip('/')}/metrics", timeout=5) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_metrics(text):
    samples = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name_and_labels, raw_value = line.rsplit(None, 1)
        match = re.fullmatch(r"([^{}]+)(?:\{(.*)\})?", name_and_labels)
        if not match:
            continue
        name, raw_labels = match.groups()
        labels = dict(re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', raw_labels or ""))
        try:
            samples[(name, tuple(sorted(labels.items())))] = float(raw_value)
        except ValueError:
            continue
    return samples


def metric_sum(samples, name, room=None, reasons=None):
    total = 0.0
    for (sample_name, raw_labels), value in samples.items():
        labels = dict(raw_labels)
        if sample_name != name:
            continue
        if room is not None and labels.get("room_id") != room:
            continue
        if reasons is not None and labels.get("reason") not in reasons:
            continue
        total += value
    return total


async def wait_for_processing_drain(url, before_text, expected_by_room, timeout):
    before = parse_metrics(before_text)
    started = time.monotonic()
    latest = {}
    while True:
        try:
            latest_text = await asyncio.to_thread(read_metrics, url)
        except Exception as exc:
            return {"complete": False, "error": f"metrics read failed: {exc}"}
        latest = parse_metrics(latest_text)
        rooms = {}
        all_complete = True
        for room, sent in expected_by_room.items():
            def delta(name, reasons=None):
                current = metric_sum(latest, name, room, reasons)
                initial = metric_sum(before, name, room, reasons)
                return max(0, int(round(current - initial)))

            received = delta("subtitle_audio_chunks_total")
            stt_done = delta("subtitle_stt_completed_chunks_total")
            stt_nonempty = delta("subtitle_stt_nonempty_chunks_total")
            eligible = delta("subtitle_translation_eligible_chunks_total")
            translated = delta("subtitle_translation_completed_chunks_total")
            short_or_filtered = delta(
                "subtitle_translation_skipped_chunks_total",
                {"hallucination", "invalid_output", "empty_output"},
            )
            empty_stt = delta("subtitle_translation_skipped_chunks_total", {"empty_stt"})
            current_depth = int(round(metric_sum(latest, "subtitle_audio_ingress_queue_depth", room)))
            complete = (
                received >= sent
                and stt_done >= received
                and stt_nonempty + empty_stt >= stt_done
                and eligible + delta("subtitle_translation_skipped_chunks_total", {"short_text"})
                    >= stt_nonempty
                and translated + short_or_filtered >= eligible
                and current_depth == 0
            )
            rooms[room] = {
                "audio_sent": sent,
                "audio_received": received,
                "stt_completed": stt_done,
                "stt_nonempty": stt_nonempty,
                "translation_eligible": eligible,
                "translation_completed": translated,
                "skipped_nonempty": short_or_filtered,
                "empty_stt": empty_stt,
                "queue_depth": current_depth,
                "complete": complete,
            }
            all_complete = all_complete and complete

        if all_complete:
            return {
                "complete": True,
                "waited_seconds": round(time.monotonic() - started, 2),
                "rooms": rooms,
            }
        if time.monotonic() - started >= timeout:
            return {
                "complete": False,
                "waited_seconds": round(time.monotonic() - started, 2),
                "rooms": rooms,
            }
        await asyncio.sleep(1)


async def professor(uri, room, src_lang, tgt_lang, chunks, chunk_seconds,
                    duration_seconds, stop_professors, sending_done, send_status):
    sent = 0
    late_chunks = 0
    max_lateness_ms = 0.0
    server_errors = []

    async def receive_server_messages(ws):
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if msg.get("type") == "error":
                    server_errors.append(msg.get("text", "unknown server error"))
        except ConnectionClosed:
            return

    try:
        async with websockets.connect(
            uri, max_size=None, ping_interval=20, ping_timeout=120
        ) as ws:
            await ws.send(json.dumps({
                "room_id": room,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }))
            receive_task = asyncio.create_task(receive_server_messages(ws))
            loop = asyncio.get_running_loop()
            started_at = loop.time()
            try:
                while True:
                    for chunk in chunks:
                        due_at = started_at + sent * chunk_seconds
                        if duration_seconds and due_at - started_at >= duration_seconds:
                            sending_done.set()
                            await stop_professors.wait()
                            return {
                                "chunks_sent": sent,
                                "late_chunks": late_chunks,
                                "max_send_lateness_ms": round(max_lateness_ms, 3),
                                "server_errors": server_errors,
                                "error": None,
                            }
                        await asyncio.sleep(max(0.0, due_at - loop.time()))
                        await ws.send(chunk)
                        lateness_ms = max(0.0, (loop.time() - due_at) * 1000)
                        if lateness_ms >= 100:
                            late_chunks += 1
                        max_lateness_ms = max(max_lateness_ms, lateness_ms)
                        sent += 1
                        send_status["chunks_sent"] = sent
                        if not duration_seconds and sent >= len(chunks):
                            sending_done.set()
                            await stop_professors.wait()
                            return {
                                "chunks_sent": sent,
                                "late_chunks": late_chunks,
                                "max_send_lateness_ms": round(max_lateness_ms, 3),
                                "server_errors": server_errors,
                                "error": None,
                            }
            finally:
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
    except Exception as exc:
        return {
            "chunks_sent": sent,
            "late_chunks": late_chunks,
            "max_send_lateness_ms": round(max_lateness_ms, 3),
            "server_errors": server_errors,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sending_done.set()


async def viewer(uri, room, results, stop, all_connected, expected_connections):
    async with websockets.connect(
        f"{uri}?room_id={room}", max_size=None, ping_interval=20, ping_timeout=120
    ) as ws:
        results["connections_by_room"][room] += 1
        results["connections_established"] += 1
        if results["connections_established"] >= expected_connections:
            all_connected.set()

        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1)
            except asyncio.TimeoutError:
                continue
            except ConnectionClosed:
                return

            try:
                msg = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if msg.get("type") != "translation" or not msg.get("done"):
                continue

            timing = msg.get("timing") or {}
            room_results = results["by_room"][room]
            if timing.get("audio_received_at") is not None:
                value = (time.time() - timing["audio_received_at"]) * 1000
                results["latencies"]["e2e_ms"].append(value)
                room_results["e2e_ms"].append(value)

            for name, start_key, end_key in (
                ("audio_to_stt", "audio_received_at", "stt_completed_at"),
                ("stt_to_queue", "stt_completed_at", "translation_queued_at"),
                ("queue_to_delivery_server", "translation_queued_at",
                 "student_delivery_started_at"),
            ):
                if start_key in timing and end_key in timing:
                    value = (timing[end_key] - timing[start_key]) * 1000
                    results["latencies"][name].append(value)
                    room_results[name].append(value)


async def run(args, chunks):
    base = args.url.rstrip("/")
    professor_uri = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/subtitle"
    viewer_uri = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/viewer"
    rooms = list(DEFAULT_TARGETS)
    expected = len(rooms) * args.students_per_room
    stop = asyncio.Event()
    stop_professors = asyncio.Event()
    all_connected = asyncio.Event()
    results = {
        "connections_established": 0,
        "connections_by_room": {room: 0 for room in rooms},
        "latencies": {name: [] for name in LATENCY_NAMES},
        "by_room": {
            room: {name: [] for name in LATENCY_NAMES} for room in rooms
        },
    }
    viewers = [
        asyncio.create_task(viewer(
            viewer_uri, room, results, stop, all_connected, expected
        ))
        for room in rooms
        for _ in range(args.students_per_room)
    ]
    try:
        metrics_before = await asyncio.to_thread(read_metrics, args.url)
        metrics_before_error = None
    except Exception as exc:
        metrics_before = None
        metrics_before_error = f"{type(exc).__name__}: {exc}"

    connection_timeout = False
    processing_status = None
    try:
        try:
            await asyncio.wait_for(all_connected.wait(), timeout=args.connect_timeout)
        except asyncio.TimeoutError:
            connection_timeout = True
            print(
                f"Warning: only {results['connections_established']}/{expected} "
                "viewer connections were established before timeout.",
                flush=True,
            )

        await asyncio.sleep(args.warmup)
        sending_done = {room: asyncio.Event() for room in rooms}
        send_status = {room: {"chunks_sent": 0} for room in rooms}
        professor_tasks = [asyncio.create_task(professor(
                professor_uri, room, args.src_lang, DEFAULT_TARGETS[room],
                chunks, args.chunk_seconds, args.duration_seconds,
                stop_professors, sending_done[room], send_status[room],
            )) for room in rooms]
        await asyncio.gather(*(event.wait() for event in sending_done.values()))
        expected_by_room = {
            room: status["chunks_sent"] for room, status in send_status.items()
        }
        if metrics_before is not None:
            processing_status = await wait_for_processing_drain(
                args.url, metrics_before, expected_by_room, args.drain
            )
        else:
            await asyncio.sleep(args.drain)
            processing_status = {
                "complete": False,
                "error": f"initial metrics unavailable: {metrics_before_error}",
            }
        stop_professors.set()
        professor_results = await asyncio.gather(*professor_tasks)
    finally:
        stop_professors.set()
        stop.set()
        await asyncio.gather(*viewers, return_exceptions=True)

    return {
        "audio": args.audio,
        "source_language": args.src_lang,
        "room_targets": DEFAULT_TARGETS,
        "students_per_room_expected": args.students_per_room,
        "viewer_connections_expected": expected,
        "viewer_connections_established": results["connections_established"],
        "viewer_connections_by_room": results["connections_by_room"],
        "connection_timeout": connection_timeout,
        "duration_seconds_requested": args.duration_seconds,
        "drain_seconds": args.drain,
        "processing_status": processing_status,
        "audio_chunks_sent_by_room": {
            room: result["chunks_sent"]
            for room, result in zip(rooms, professor_results)
        },
        "late_audio_chunks_by_room": {
            room: result["late_chunks"]
            for room, result in zip(rooms, professor_results)
        },
        "max_send_lateness_ms_by_room": {
            room: result["max_send_lateness_ms"]
            for room, result in zip(rooms, professor_results)
        },
        "professor_errors_by_room": {
            room: result["error"] for room, result in zip(rooms, professor_results)
        },
        "server_errors_by_room": {
            room: result["server_errors"] for room, result in zip(rooms, professor_results)
        },
        "latency": {
            name: summary(results["latencies"][name]) for name in LATENCY_NAMES
        },
        "room_latency": {
            room: {
                name: summary(values[name]) for name in LATENCY_NAMES
            }
            for room, values in results["by_room"].items()
        },
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--audio", default="wav/brian-neutral-and-measured.wav",
        help="shared mono 16 kHz, 16-bit PCM WAV for all three rooms",
    )
    parser.add_argument(
        "--src-lang", default="en",
        help="source language code (default: en; use ko for Korean speech)",
    )
    parser.add_argument("--students-per-room", type=int, default=100)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument(
        "--duration-seconds", type=float, default=0,
        help="repeat the shared WAV at real-time chunk cadence for this long; 0 sends it once",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument(
        "--drain", type=float, default=90.0,
        help="maximum seconds to wait for all audio/STT/translation metrics to drain",
    )
    parser.add_argument("--output", default="multilingual-load-test-results.json")
    args = parser.parse_args()

    if args.students_per_room < 1:
        parser.error("--students-per-room must be at least 1")
    if args.chunk_seconds <= 0 or args.connect_timeout <= 0 or args.duration_seconds < 0:
        parser.error("--chunk-seconds and --connect-timeout must be positive; duration cannot be negative")

    chunks = read_audio(args.audio, args.chunk_seconds)
    report = await run(args, chunks)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
