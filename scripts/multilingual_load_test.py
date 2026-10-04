#!/usr/bin/env python3
"""Send one WAV concurrently to Rooms 1-3 with distinct target languages."""
import argparse
import asyncio
import json
import time

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


async def professor(uri, room, src_lang, tgt_lang, chunks, chunk_seconds):
    async with websockets.connect(uri, max_size=None, ping_interval=20) as ws:
        await ws.send(json.dumps({
            "room_id": room,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }))
        for chunk in chunks:
            await ws.send(chunk)
            await asyncio.sleep(chunk_seconds)


async def viewer(uri, room, results, stop, all_connected, expected_connections):
    async with websockets.connect(
        f"{uri}?room_id={room}", max_size=None, ping_interval=20
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

    connection_timeout = False
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
        professors = [
            asyncio.create_task(professor(
                professor_uri, room, args.src_lang, DEFAULT_TARGETS[room],
                chunks, args.chunk_seconds,
            ))
            for room in rooms
        ]
        await asyncio.gather(*professors)
        await asyncio.sleep(args.drain)
    finally:
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
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--drain", type=float, default=30.0)
    parser.add_argument("--output", default="multilingual-load-test-results.json")
    args = parser.parse_args()

    if args.students_per_room < 1:
        parser.error("--students-per-room must be at least 1")
    if args.chunk_seconds <= 0 or args.connect_timeout <= 0:
        parser.error("--chunk-seconds and --connect-timeout must be positive")

    chunks = read_audio(args.audio, args.chunk_seconds)
    report = await run(args, chunks)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
