import io
import json
import struct
import time
import numpy as np
import librosa
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.diagnosis_pipeline import analyze_note
from app.models.schemas import NoteCaptureMeta
from pytune_dsp.utils.note_utils import midi_to_freq

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
                meta_bytes = data[4:4 + meta_len]
                audio_bytes = data[4 + meta_len:]

                meta_dict = json.loads(meta_bytes.decode("utf-8"))
                meta = NoteCaptureMeta(**meta_dict)

                print(f"🎹 Received note {meta.note_expected}, {len(audio_bytes)} bytes")

                # 2) Essayer décodage direct PCM (rapide)
                signal = None
                sr = meta.sample_rate
                try:
                    signal = np.frombuffer(audio_bytes, dtype=np.float32)

                    # Gestion des canaux
                    if meta.channels == 2:
                        signal = signal.reshape(-1, 2).T  # (2, N)
                    elif meta.channels == 1:
                        pass
                    else:
                        print(f"⚠️ Channels non supportés: {meta.channels}")

                except Exception as e:
                    print(f"⚠️ PCM decode failed, trying librosa... ({e})")
                    try:
                        audio_buf = io.BytesIO(audio_bytes)
                        signal, _ = librosa.load(audio_buf, sr=sr, mono=False)
                    except Exception as e2:
                        print(f"❌ Impossible de décoder l’audio: {e2}")
                        continue

                # Si stéréo → mixer en mono
                if signal.ndim > 1:
                    signal = np.mean(signal, axis=0)

                # 3) Analyse via pipeline
                expected_freq = midi_to_freq(meta.note_expected)
                result = analyze_note(str(meta.note_expected), expected_freq, signal, sr)

                # 4) Réponse enrichie
                payload = {
                    "type": "analysis",
                    "note": meta.note_expected,
                    **json.loads(result.model_dump_json())
                }
                await ws.send_text(json.dumps(payload))
                continue

            if msg["type"] == "websocket.disconnect":
                print("❌ Client disconnected from /diag/ws")
                break

    except WebSocketDisconnect:
        print("❌ Client disconnected from /diag/ws")