#!/usr/bin/env python3
"""Keep virtual student WebSocket viewers connected without sending audio."""
import argparse
import asyncio
import json
from urllib.parse import urlencode, urlsplit, urlunsplit

import websockets
from websockets.exceptions import ConnectionClosed


def viewer_url(base_url: str, room: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        raise ValueError("--url must be an absolute http(s) or ws(s) URL")
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}[parsed.scheme]
    path = parsed.path.rstrip("/") + "/ws/viewer"
    return urlunsplit((scheme, parsed.netloc, path, urlencode({"room_id": room}), ""))


async def keep_viewer(
    url: str,
    room: str,
    viewer_id: int,
    stats: dict,
    stop: asyncio.Event,
    stagger_seconds: float,
) -> None:
    if stagger_seconds:
        await asyncio.sleep(viewer_id * stagger_seconds)

    connected_once = False
    retry_delay = 1.0
    while not stop.is_set():
        socket_active = False
        try:
            async with websockets.connect(
                url,
                max_size=None,
                max_queue=32,
                open_timeout=15,
                ping_interval=20,
                ping_timeout=20,
            ) as socket:
                stats["active"][room] += 1
                socket_active = True
                retry_delay = 1.0
                if not connected_once:
                    connected_once = True
                    stats["connected_once"] += 1
                    total = stats["expected"]
                    if stats["connected_once"] % 25 == 0 or stats["connected_once"] == total:
                        print(
                            f"뷰어 연결 {stats['connected_once']}/{total} | "
                            + " ".join(
                                f"Room {r}: {stats['active'][r]}" for r in stats["rooms"]
                            ),
                            flush=True,
                        )

                while not stop.is_set():
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=1)
                    except asyncio.TimeoutError:
                        continue
                    except ConnectionClosed:
                        break
                    try:
                        message = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if message.get("type") == "translation" and message.get("done"):
                        stats["translations"][room] += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not stop.is_set():
                print(f"Room {room} viewer {viewer_id} 연결 오류: {exc}", flush=True)
        finally:
            if socket_active:
                # This task contributes at most one active socket at a time.
                stats["active"][room] -= 1

        if not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=retry_delay)
            except asyncio.TimeoutError:
                retry_delay = min(retry_delay * 2, 5.0)


def parse_rooms(value: str) -> list[str]:
    rooms = [room.strip() for room in value.split(",") if room.strip()]
    if not rooms or any(room not in {"1", "2", "3", "4", "5"} for room in rooms):
        raise argparse.ArgumentTypeError("rooms must be unique IDs from 1 to 5")
    if len(set(rooms)) != len(rooms):
        raise argparse.ArgumentTypeError("room IDs must not be repeated")
    return rooms


async def run(args) -> None:
    rooms = args.rooms
    urls = {room: viewer_url(args.url, room) for room in rooms}
    expected = len(rooms) * args.students_per_room
    stats = {
        "rooms": rooms,
        "expected": expected,
        "connected_once": 0,
        "active": {room: 0 for room in rooms},
        "translations": {room: 0 for room in rooms},
    }
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(
            keep_viewer(
                urls[room], room, index, stats, stop, args.stagger_ms / 1000
            )
        )
        for room in rooms
        for index in range(args.students_per_room)
    ]

    print(
        f"Room {','.join(rooms)}에 룸당 {args.students_per_room}명 "
        f"(총 {expected}명) 연결 중. 교수 오디오는 전송하지 않습니다.",
        flush=True,
    )
    try:
        await asyncio.gather(*tasks)
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("\n가상 학생 연결을 종료했습니다.", flush=True)
        print(
            "완료 번역 수신: "
            + " | ".join(
                f"Room {room}: {stats['translations'][room]}"
                for room in rooms
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--rooms", type=parse_rooms, default=parse_rooms("1,2,3"),
        help="comma-separated room IDs",
    )
    parser.add_argument("--students-per-room", type=int, default=100)
    parser.add_argument(
        "--stagger-ms", type=float, default=5.0,
        help="delay between opening viewer connections (default: 5 ms)",
    )
    args = parser.parse_args()
    if args.students_per_room < 1:
        parser.error("--students-per-room must be at least 1")
    if args.stagger_ms < 0:
        parser.error("--stagger-ms cannot be negative")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n중단 요청을 받았습니다.", flush=True)


if __name__ == "__main__":
    main()
