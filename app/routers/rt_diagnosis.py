import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse
import asyncio
from fastapi import APIRouter, WebSocket
import time

router = APIRouter(prefix="/diag")

@router.websocket("/ws/diagnosis")
async def diagnosis_ws(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            msg = await ws.receive()
            print("📩 Message brut reçu:", msg)

            if msg["type"] == "websocket.receive":
                data = msg.get("text") or msg.get("bytes")
                print("➡️ Data:", data)

                if data == "ping":
                    start = time.time()
                    await ws.send_text("pong")
                    latency = (time.time() - start) * 1000
                    await ws.send_text(f"latency:{latency:.2f}ms")

            elif msg["type"] == "websocket.disconnect":
                print("❌ Client déconnecté")
                break

        except Exception as e:
            import traceback
            print("❌ Exception WebSocket:", e)
            traceback.print_exc()
            break



@router.websocket("/ws")
async def ws_diagnosis(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 🔹 recevoir un buffer audio (ArrayBuffer côté frontend)
            data = await websocket.receive_bytes()

            # pour l’instant on ne fait que confirmer réception
            print(f"📦 Audio chunk reçu: {len(data)} bytes")

            # plus tard : appel de ta fonction d’analyse (yin/fft, etc.)
            result = {
                "note_detected": None,
                "frequency_hz": None,
                "confidence": None,
                "received_bytes": len(data)
            }

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("❌ Client déconnecté du diagnostic")