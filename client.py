"""
실시간 자막 WebSocket 클라이언트
----------------------------------
마이크 입력을 캡처해서 server.py 로 전송하고 번역 결과를 출력합니다.

사용법:
  python client.py
  python client.py --src en --tgt ko
  python client.py --url ws://192.168.0.10:8000/ws/subtitle
"""

import argparse
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import websocket

# ── 설정 ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
CHUNK_SAMPLES    = 512
VAD_THRESHOLD    = 0.02
SILENCE_DURATION = 0.7   # 초


# ── VAD 기반 오디오 캡처 (stt_translate_test.py 와 동일) ──────────────────────
def audio_capture_loop(speech_queue: queue.Queue, stop_event: threading.Event):
    raw_q = queue.Queue()
    silence_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SAMPLES)

    def callback(indata, frames, time_info, status):
        raw_q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK_SAMPLES, callback=callback):
        print("[마이크] 켜짐, 말하면 됩니다.", flush=True)
        while not stop_event.is_set():
            speaking      = False
            silence_count = 0
            audio_chunks  = []

            while not stop_event.is_set():
                try:
                    chunk = raw_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > VAD_THRESHOLD:
                    if not speaking:
                        speaking = True
                    silence_count = 0
                    audio_chunks.append(chunk)
                elif speaking:
                    audio_chunks.append(chunk)
                    silence_count += 1
                    if silence_count >= silence_limit:
                        break

            if audio_chunks:
                audio = np.concatenate(audio_chunks, axis=0).flatten()
                speech_queue.put(audio)


# ── WebSocket 클라이언트 ───────────────────────────────────────────────────────

def recv_loop(ws: websocket.WebSocket, stop_event: threading.Event):
    """수신 전용 스레드: 서버 응답을 계속 받아서 출력"""
    while not stop_event.is_set():
        try:
            msg  = ws.recv()
            data = json.loads(msg)
            if data["type"] == "stt":
                print(f"[원문]  {data['text']}", flush=True)
            elif data["type"] == "translation":
                print(f"[번역]  {data['text']}\n", flush=True)
            elif data["type"] == "error":
                print(f"[오류]  {data['text']}", flush=True)
        except Exception:
            break


def run(server_url: str, src_lang: str, tgt_lang: str):
    speech_queue = queue.Queue()
    stop_event   = threading.Event()

    ws = websocket.create_connection(server_url)
    print(f"[서버 연결] {server_url}")
    print(f"[번역 방향] {src_lang} → {tgt_lang}")
    print("Ctrl+C 로 종료\n")

    # 언어 설정 전송 (1회)
    ws.send(json.dumps({"src_lang": src_lang, "tgt_lang": tgt_lang}))

    # 수신 스레드 시작 (송신과 독립적으로 응답 출력)
    recv_thread = threading.Thread(target=recv_loop, args=(ws, stop_event), daemon=True)
    recv_thread.start()

    # 마이크 캡처 스레드 시작
    audio_thread = threading.Thread(
        target=audio_capture_loop, args=(speech_queue, stop_event), daemon=True
    )
    audio_thread.start()

    try:
        while True:
            try:
                audio = speech_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 오디오를 float32 바이너리로 즉시 전송 (응답 대기 없음)
            ws.send_binary(audio.astype(np.float32).tobytes())

    except KeyboardInterrupt:
        print("\n종료합니다.")
        stop_event.set()
        ws.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="ko",                            help="입력 언어 (기본: ko)")
    parser.add_argument("--tgt", default="en",                            help="출력 언어 (기본: en)")
    parser.add_argument("--url", default="ws://localhost:8000/ws/subtitle", help="서버 WebSocket URL")
    args = parser.parse_args()
    run(args.url, args.src, args.tgt)


if __name__ == "__main__":
    main()
