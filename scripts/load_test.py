#!/usr/bin/env python3
"""Realtime WebSocket load test. WAV input must be mono PCM at 16 kHz."""
import argparse
import asyncio
import json
import math
import statistics
import struct
import time
import wave

import websockets
from websockets.exceptions import ConnectionClosed


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)], 3)


def summary(values):
    if not values:
        return {"count": 0, "avg_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {"count": len(values), "avg_ms": round(statistics.fmean(values), 3),
            "p50_ms": percentile(values, .50), "p95_ms": percentile(values, .95),
            "p99_ms": percentile(values, .99)}


def read_audio(path, chunk_seconds):
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError("WAV must be mono, 16 kHz, 16-bit PCM")
        frames_per_chunk = max(1, int(16000 * chunk_seconds))
        chunks = []
        while raw := wav.readframes(frames_per_chunk):
            n = len(raw) // 2
            chunks.append(struct.pack(f"={n}f", *(x / 32768.0 for x in struct.unpack(f"={n}h", raw))))
        if not chunks:
            raise ValueError("WAV contains no audio")
        return chunks


async def room_professor(uri, room, chunks, chunk_seconds):
    async with websockets.connect(uri, max_size=None, ping_interval=20) as ws:
        await ws.send(json.dumps({"room_id": str(room), "src_lang": "ko", "tgt_lang": "en"}))
        for chunk in chunks:
            await ws.send(chunk)
            await asyncio.sleep(chunk_seconds)


async def viewer(uri, room, results, stop):
    async with websockets.connect(f"{uri}?room_id={room}", max_size=None, ping_interval=20) as ws:
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
            # The load runner and API host are expected to share the local clock. This gives
            # a per-viewer transport-inclusive latency; server stage timings are also retained.
            if timing.get("audio_received_at") is not None:
                results["e2e_ms"].append((time.time() - timing["audio_received_at"]) * 1000)
            for name, start_key, end_key in (
                ("audio_to_stt", "audio_received_at", "stt_completed_at"),
                ("stt_to_queue", "stt_completed_at", "translation_queued_at"),
                ("queue_to_delivery_server", "translation_queued_at", "student_delivery_started_at"),
            ):
                if start_key in timing and end_key in timing:
                    results[name].append((timing[end_key] - timing[start_key]) * 1000)


async def run_phase(args, room_count, room_chunks):
    base = args.url.rstrip("/")
    professor_uri = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/subtitle"
    viewer_uri = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/viewer"
    rooms = list(range(1, room_count + 1))
    stop = asyncio.Event()
    results = {"e2e_ms": [], "audio_to_stt": [],
               "stt_to_queue": [], "queue_to_delivery_server": []}
    viewers = [asyncio.create_task(viewer(viewer_uri, room, results, stop))
               for room in rooms for _ in range(args.students_per_room)]
    await asyncio.sleep(args.warmup)
    professors = [asyncio.create_task(room_professor(professor_uri, room, room_chunks[room],
                                                      args.chunk_seconds))
                  for room in rooms]
    await asyncio.gather(*professors)
    await asyncio.sleep(args.drain)
    stop.set()
    await asyncio.gather(*viewers, return_exceptions=True)
    return {"rooms": room_count, "viewer_connections_expected": room_count * args.students_per_room,
            "latency": {name: summary(results[name]) for name in
                        ("e2e_ms", "audio_to_stt", "stt_to_queue", "queue_to_delivery_server")}}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--audio", help="default mono 16 kHz 16-bit PCM WAV for rooms without an override")
    parser.add_argument("--room-audio", action="append", default=[], metavar="ROOM=FILE",
                        help="WAV file for a room; repeat to assign different audio (e.g. --room-audio 1=a.wav)")
    parser.add_argument("--rooms", default="1,3,5", help="comma-separated phases (maximum 5)")
    parser.add_argument("--students-per-room", type=int, default=50)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--drain", type=float, default=30.0)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output", default="load-test-results.json")
    args = parser.parse_args()
    try:
        phases = [int(x) for x in args.rooms.split(",")]
    except ValueError:
        parser.error("--rooms must be a comma-separated list of integers")
    if any(room_count < 1 or room_count > 5 for room_count in phases):
        parser.error("--rooms must contain values from 1 to 5")
    if not phases:
        parser.error("--rooms must contain at least one phase")

    room_audio = {}
    for assignment in args.room_audio:
        room, separator, path = assignment.partition("=")
        if not separator or room not in {"1", "2", "3", "4", "5"} or not path:
            parser.error(f"invalid --room-audio {assignment!r}; expected ROOM=FILE with ROOM from 1 to 5")
        if room in room_audio:
            parser.error(f"audio for room {room} was assigned more than once")
        room_audio[room] = path

    max_rooms = max(phases)
    if not args.audio and any(str(room) not in room_audio for room in range(1, max_rooms + 1)):
        parser.error("provide --audio as a default or assign --room-audio for every room used by --rooms")
    room_chunks = {}
    for room in range(1, max_rooms + 1):
        path = room_audio.get(str(room), args.audio)
        room_chunks[room] = read_audio(path, args.chunk_seconds)
    reports = []
    for room_count in phases:
        print(f"Starting {room_count} Room phase ({room_count * args.students_per_room} viewers)", flush=True)
        report = await run_phase(args, room_count, room_chunks)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if room_count != phases[-1]:
            await asyncio.sleep(args.cooldown)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"phases": reports}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
