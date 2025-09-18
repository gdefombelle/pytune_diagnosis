import json
import struct
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.note_analyzer import analyze_expected_note
from app.models.schemas import NoteCaptureMeta

router = APIRouter(prefix="/diag")

@router.websocket("/ws")
async def ws_diagnosis(ws: WebSocket):
    await ws.accept()
    print("✅ Client connected to /diag/ws")

    try:
        while True:
            msg = await ws.receive()

            # ────────────── PING / PONG ──────────────
            if msg["type"] == "websocket.receive" and "text" in msg:
                text = msg["text"]
                if text == "ping":
                    start = time.time()
                    await ws.send_text("pong")
                    latency = (time.time() - start) * 1000
                    await ws.send_text(f"latency:{latency:.2f}ms")
                    continue

            # ────────────── ANALYSE AUDIO ──────────────
            if msg["type"] == "websocket.receive" and "bytes" in msg:
                data = msg["bytes"]

                # 1) Lire header JSON
                meta_len = struct.unpack("I", data[:4])[0]
                meta_bytes = data[4:4+meta_len]
                audio_bytes = data[4+meta_len:]

                meta_dict = json.loads(meta_bytes.decode("utf-8"))
                meta = NoteCaptureMeta(**meta_dict)

                print(f"🎹 Received note {meta.note_expected}, {len(audio_bytes)} bytes")

                # 2) Analyse
                result = analyze_expected_note(meta, audio_bytes)

                # 3) Réponse enrichie
                payload = {
                    "type": "analysis",
                    "note": meta.note_expected,
                    **json.loads(result.json())
                }
                await ws.send_text(json.dumps(payload))
                continue

            if msg["type"] == "websocket.disconnect":
                print("❌ Client disconnected from /diag/ws")
                break

    except WebSocketDisconnect:
        print("❌ Client disconnected from /diag/ws")