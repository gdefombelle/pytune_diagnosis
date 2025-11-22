# app/routers/diagnosis_router.py
import io
import json
import struct
import time
import numpy as np
import librosa
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pytune_dsp.types.schemas import NoteAnalysisResult
from pytune_dsp.utils.note_utils import midi_to_freq
from pytune_dsp.utils.pianos import infer_era_from_year
from app.core.diagnosis_pipeline import analyze_note

router = APIRouter(prefix="/diag")

import numpy as np
import json

def safe_json_default(obj):
    """Convertit proprement les objets non sérialisables en JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    return str(obj)

@router.websocket("/ws")
async def ws_diagnosis(ws: WebSocket):
    await ws.accept()
    print("✅ Client connected to /diag/ws")

    try:
        while True:
            msg = await ws.receive()

            # --- PING/PONG ---
            if msg["type"] == "websocket.receive" and "text" in msg:
                text = msg["text"]
                if text == "ping":
                    start = time.time()
                    await ws.send_text("pong")
                    latency = (time.time() - start) * 1000
                    await ws.send_text(f"latency:{latency:.2f}ms")
                    continue

            # --- AUDIO ---
            if msg["type"] == "websocket.receive" and "bytes" in msg:
                data = msg["bytes"]

                # 1️⃣ Extraire header JSON + données audio
                meta_len = struct.unpack("I", data[:4])[0]
                meta_bytes = data[4:4 + meta_len]
                audio_bytes = data[4 + meta_len:]

                meta = json.loads(meta_bytes.decode("utf-8"))
                note_expected = meta["note_expected"]
                expected_freq = midi_to_freq(note_expected)

                # --- Infos piano transmises ---
                piano_meta = meta.get("piano", {}) or {}
                piano_id = piano_meta.get("id")
                piano_type = (piano_meta.get("type") or "upright").lower()
                piano_era = (piano_meta.get("era") or "").lower()

                if not piano_era and piano_meta.get("year"):
                    piano_era = infer_era_from_year(piano_meta["year"])

                if piano_type not in ("upright", "grand"):
                    piano_type = "upright"
                if piano_era not in ("antique", "vintage", "modern"):
                    piano_era = "modern"

                streams_meta = meta.get("streams")

                # --- Extraction des signaux audio ---
                signals = []
                if streams_meta:
                    offset = 0
                    for s_meta in streams_meta:
                        sr = s_meta.get("sample_rate", meta.get("sample_rate", 48000))
                        length = s_meta["length"]
                        raw = audio_bytes[offset: offset + length * 4]
                        offset += length * 4
                        sig = np.frombuffer(raw, dtype=np.float32)
                        signals.append((sr, sig, s_meta))
                else:
                    sr = meta.get("sample_rate", 48000)
                    sig = np.frombuffer(audio_bytes, dtype=np.float32)
                    signals.append((sr, sig, {"stream": 0}))

                if not signals:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "error": "No valid audio streams"
                    }))
                    continue

                # 2️⃣ Analyse de chaque flux
                results = []
                for i, (sr, sig, meta_stream) in enumerate(signals):
                    dur = len(sig) / sr
                    try:
                        res: NoteAnalysisResult = analyze_note(
                            note_name=str(note_expected),
                            expected_freq=expected_freq,
                            signal=sig,
                            sr=sr,
                            compute_inharm=meta.get("compute_inharm", False),
                            piano_type=piano_type,
                            era=piano_era,
                            use_essentia=True,          # si on veut garder la voie "informed"
                            use_librosa=False,          # tu peux laisser False
                            use_yin_partitioned=True,   # <── ON active le nouveau moteur
                            yin_partition_mode="cents_250",  # ou "octaves_8", "cents_250"
                            yin_target_width_oct=1.0,
                            yin_max_depth=5,
                        )
                        results.append({
                            "stream_index": i,
                            "duration_s": dur,
                            "samples": len(sig),
                            "sample_rate": sr,
                            "analysis": res.model_dump(),
                        })
                    except Exception as e:
                        results.append({
                            "stream_index": i,
                            "error": str(e),
                        })

                # 3️⃣ Sélection du meilleur flux (confiance max)
                best, best_score = None, -1.0
                for r in results:
                    a = r.get("analysis")
                    if not a:
                        continue
                    g = a.get("guessed_note")
                    conf = g.get("confidence", 0.0) if g else 0.0
                    if conf > best_score:
                        best_score = conf
                        best = a

                # 4️⃣ Construction du payload de réponse
                payload = {
                    "type": "analysis",
                    "midi": note_expected,
                    "noteName": best.get("note_name") if best else None,
                    **(best or {}),
                    "piano": {
                        "id": piano_id,
                        "type": piano_type,
                        "era": piano_era,
                    },
                    "streams_debug": results,  # ⚠️ retirer en production
                }

                await ws.send_text(json.dumps(payload, default=safe_json_default))
                continue

            if msg["type"] == "websocket.disconnect":
                print("❌ Client disconnected from /diag/ws")
                break

    except WebSocketDisconnect as e:
        print(f"❌ Client disconnected from /diag/ws -exception: {e}")