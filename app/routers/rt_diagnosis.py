# app/routers/diagnosis_router.py
import asyncio
import json
import struct
import time
import wave
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pytune_dsp.types.schemas import NoteAnalysisResult
from pytune_dsp.utils.note_utils import midi_to_freq
from pytune_dsp.utils.pianos import infer_era_from_year

from app.core.diagnosis_pipeline import analyze_note
from app.services.sse_publisher import publish_event
from app.models.models import slim_note_analysis

from pytune_data.minio_client import (
    minio_client,
    PYTUNE_DIAGNOSIS_AUDIO_BUCKET,
)

router = APIRouter(prefix="/diag")

import numpy as np
import json

_DIAG_AUDIO_BUCKET_READY = False

def _ensure_diag_audio_bucket() -> None:
    global _DIAG_AUDIO_BUCKET_READY
    if _DIAG_AUDIO_BUCKET_READY:
        return

    try:
        if not minio_client.client.bucket_exists(PYTUNE_DIAGNOSIS_AUDIO_BUCKET):
            minio_client.client.make_bucket(PYTUNE_DIAGNOSIS_AUDIO_BUCKET)
    except Exception:
        # au cas où un autre worker/process l'ait créé juste avant
        if not minio_client.client.bucket_exists(PYTUNE_DIAGNOSIS_AUDIO_BUCKET):
            raise

    _DIAG_AUDIO_BUCKET_READY = True


def _float32_to_wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    """
    Convertit un signal mono float32 [-1..1] en WAV PCM16.
    """
    sig = np.asarray(signal, dtype=np.float32)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig = np.clip(sig, -1.0, 1.0)

    pcm16 = (sig * 32767.0).astype(np.int16)

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # PCM16
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())

    return buf.getvalue()


def _build_diag_audio_object_key(
    session_id: int | None,
    midi: int,
    stream_index: int,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    session_part = f"session_{session_id}" if session_id is not None else "session_unknown"
    return f"diagnosis/{session_part}/notes/midi_{midi}/stream_{stream_index}_{ts}.wav"


def _upload_diag_audio_sync(
    *,
    session_id: int | None,
    midi: int,
    stream_index: int,
    signal: np.ndarray,
    sample_rate: int,
    diagnosis_mode: str,
    sequence_policy: str,
    piano_meta: dict,
    meta_stream: dict,
) -> dict:
    """
    Fonction synchrone exécutée dans un thread via asyncio.to_thread().
    """
    _ensure_diag_audio_bucket()

    wav_bytes = _float32_to_wav_bytes(signal, sample_rate)
    object_key = _build_diag_audio_object_key(session_id, midi, stream_index)

    minio_client.client.put_object(
        bucket_name=PYTUNE_DIAGNOSIS_AUDIO_BUCKET,
        object_name=object_key,
        data=BytesIO(wav_bytes),
        length=len(wav_bytes),
        content_type="audio/wav",
    )

    return {
        "audio_object_key": object_key,
        "audio_bucket": PYTUNE_DIAGNOSIS_AUDIO_BUCKET,
        "audio_format": "wav",
        "audio_sample_rate": int(sample_rate),
        "audio_num_samples": int(len(signal)),
        "audio_duration_s": float(len(signal) / sample_rate) if sample_rate else None,
        "audio_channels": 1,
        "audio_meta": {
            "stream_index": stream_index,
            "diagnosis_mode": diagnosis_mode,
            "sequence_policy": sequence_policy,
            "piano_id": piano_meta.get("id"),
            "piano_type": piano_meta.get("type"),
            "piano_era": piano_meta.get("era"),
            "meta_stream": meta_stream or {},
        },
    }


async def _upload_diag_audio_batch_background(
    *,
    session_id: int | None,
    midi: int,
    diagnosis_mode: str,
    sequence_policy: str,
    piano_meta: dict,
    signals: list[tuple[int, np.ndarray, dict]],
) -> None:
    """
    Upload en arrière-plan, sans bloquer la réponse WS.
    """
    try:
        for i, (sr, sig, meta_stream) in enumerate(signals):
            await asyncio.to_thread(
                _upload_diag_audio_sync,
                session_id=session_id,
                midi=midi,
                stream_index=i,
                signal=sig,
                sample_rate=sr,
                diagnosis_mode=diagnosis_mode,
                sequence_policy=sequence_policy,
                piano_meta=piano_meta,
                meta_stream=meta_stream,
            )
        print(f"✅ Uploaded diagnosis audio samples for session={session_id}, midi={midi}")
    except Exception as e:
        print(f"⚠️ Failed to upload diagnosis audio samples: {e}")

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
                meta_len = struct.unpack("<I", data[:4])[0]
                meta_bytes = data[4:4 + meta_len]
                audio_bytes = data[4 + meta_len:]

                meta = json.loads(meta_bytes.decode("utf-8"))
                note_expected = meta["note_expected"]
                expected_freq = midi_to_freq(note_expected)
                diagnosis_mode = (meta.get("diagnosis_mode") or "full_chromatic").lower()
                sequence_policy = (meta.get("sequence_policy") or "strict_order").lower()
                # --- Choix stratégie F0 selon le mode de diag ---
                use_partitioned = sequence_policy in ("guided_flexible", "free_sampling")
                use_informed = not use_partitioned

                print(
                    f"🎯 diagnosis_mode={diagnosis_mode} | "
                    f"sequence_policy={sequence_policy} | "
                    f"use_informed_expected={use_informed} | "
                    f"use_uninformed_partitioned={use_partitioned}"
                )

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
                        sig = np.frombuffer(raw, dtype=np.float32).copy()
                        signals.append((sr, sig, s_meta))
                else:
                    sr = meta.get("sample_rate", 48000)
                    sig = np.frombuffer(audio_bytes, dtype=np.float32).copy()
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
                            use_informed_expected=use_informed,
                            use_uninformed_partitioned=use_partitioned,
                            yin_partition_mode="cents_250",
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

                # 4️⃣ Construction du payload de réponse (SLIM)
                best_slim = slim_note_analysis(best) if best else None
                # payload = {
                #     "type": "analysis",
                #     "midi": note_expected,
                #     "noteName": best.get("note_name") if best else None,
                #     **(best or {}),
                #     "piano": {
                #         "id": piano_id,
                #         "type": piano_type,
                #         "era": piano_era,
                #     },
                #     "streams_debug": results,  # ⚠️ retirer en production
                # }

                payload = {
                    "type": "analysis",
                    "midi": note_expected,
                    "noteName": best_slim.get("note_name") if best_slim else None,
                    **(best_slim or {}),
                    "piano": {
                        "id": piano_id,
                        "type": piano_type,
                        "era": piano_era,
                    },
                    # ⚠️ debug uniquement — à supprimer en prod
                    # "streams_debug": results,
                }

                await ws.send_text(json.dumps(payload, default=safe_json_default))
                # 🔥 Push event to SSE live stream too for slave remote clients
                # 📌 on flatten l’event :
                event = {
                    "type": "analysis",
                    "session_id": meta.get("sessionId"),
                    "midi": note_expected,
                    "payload": payload   # ⬅️ intègre tout dedans
                }
                await publish_event(event)
                asyncio.create_task(
                    _upload_diag_audio_batch_background(
                        session_id=meta.get("sessionId"),
                        midi=note_expected,
                        diagnosis_mode=diagnosis_mode,
                        sequence_policy=sequence_policy,
                        piano_meta={
                            "id": piano_id,
                            "type": piano_type,
                            "era": piano_era,
                        },
                        signals=signals,
                    )
                )
                continue

            if msg["type"] == "websocket.disconnect":
                print("❌ Client disconnected from /diag/ws")
                break

    except WebSocketDisconnect as e:
        print(f"❌ Client disconnected from /diag/ws -exception: {e}")


