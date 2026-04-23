import argparse
import re
import wave
from io import BytesIO
from typing import Optional

import numpy as np

from pytune_data.minio_client import (
    minio_client,
    PYTUNE_DIAGNOSIS_AUDIO_BUCKET,
)
from pytune_dsp.analysis.detect_f0_seed import detect_f0_seed
from pytune_dsp.types.dataclasses import AudioChannelInput


OBJECT_KEY_RE = re.compile(
    r"diagnosis/session_(?P<session_id>\d+|unknown)/notes/midi_(?P<midi>\d+)/(?P<filename>.+)\.wav$"
)

def pretty_candidate(c: dict | None) -> str:
    if not c:
        return "None"
    note = c.get("refined_note")
    f0 = c.get("refined_f0")
    score = c.get("refined_score")
    return f"{note} | f0={f0:.4f} Hz | score={score:.4f}" if f0 is not None and score is not None else str(c)


def print_channel_summary(summary: dict) -> None:
    print()
    print("===== CHANNEL SUMMARY (HUMAN READABLE) =====")

    print()
    print("Top candidates YIN:")
    print(f"  {summary.get('candidate_top_notes')}")

    print()
    print("FFT band selected:")
    print(f"  note   : {summary.get('fft_band_selected')}")
    print(f"  f0     : {summary.get('fft_band_selected_f0')}")
    print(f"  score  : {summary.get('fft_band_selected_score')}")

    hps = summary.get("hps_info", {}) or {}
    print()
    print("HPS global:")
    print(f"  hps_note       : {hps.get('hps_note')}")
    print(f"  hps_multi_note : {hps.get('hps_multi_note')}")
    print(f"  hps_f0         : {hps.get('hps_f0')}")
    print(f"  hps_score      : {hps.get('hps_score')}")

    print()
    print("Decision chain:")
    print(f"  chosen   : {pretty_candidate(summary.get('chosen'))}")
    print(f"    method : {summary.get('chosen_method')}")
    print(f"  chosen2  : {pretty_candidate(summary.get('chosen2'))}")
    print(f"    method : {summary.get('chosen_method_2')}")
    print(f"  chosen3  : {pretty_candidate(summary.get('chosen3'))}")
    print(f"    method : {summary.get('chosen_method_3')}")
    print(f"  chosen4  : {pretty_candidate(summary.get('chosen4'))}")
    print(f"    method : {summary.get('chosen_method_4')}")

def parse_object_key(object_key: str) -> tuple[Optional[int], int]:
    m = OBJECT_KEY_RE.match(object_key)
    if not m:
        raise ValueError(f"Object key does not match expected format: {object_key}")
    session_raw = m.group("session_id")
    expected_midi = int(m.group("midi"))
    session_id = None if session_raw == "unknown" else int(session_raw)
    return session_id, expected_midi


def list_session_audio_keys(session_id: int) -> list[str]:
    prefix = f"diagnosis/session_{session_id}/notes/"
    objects = minio_client.client.list_objects(
        PYTUNE_DIAGNOSIS_AUDIO_BUCKET,
        prefix=prefix,
        recursive=True,
    )
    keys = [obj.object_name for obj in objects if obj.object_name.endswith(".wav")]
    return sorted(keys)


def read_wav_from_minio(object_key: str) -> tuple[int, np.ndarray]:
    obj = minio_client.client.get_object(PYTUNE_DIAGNOSIS_AUDIO_BUCKET, object_key)
    try:
        raw = obj.read()
    finally:
        obj.close()
        obj.release_conn()

    with wave.open(BytesIO(raw), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if n_channels != 1:
        raise ValueError(f"Expected mono WAV, got {n_channels} channels for {object_key}")
    if sampwidth != 2:
        raise ValueError(f"Expected PCM16 WAV, got sample width={sampwidth} for {object_key}")

    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return sample_rate, signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--midi", type=int, required=True)
    parser.add_argument("--trace-level", type=str, default="summary", choices=["none", "summary", "full"])
    args = parser.parse_args()

    keys = list_session_audio_keys(args.session_id)
    target = None
    for k in keys:
        _, midi = parse_object_key(k)
        if midi == args.midi:
            target = k
            break

    if target is None:
        raise RuntimeError(f"No WAV found for session={args.session_id}, midi={args.midi}")

    sr, signal = read_wav_from_minio(target)

    channels = [
        AudioChannelInput(
            channel_id="ch_0",
            signal=signal,
            sample_rate=sr,
        )
    ]

    result = detect_f0_seed(
        channels,
        trace_level=args.trace_level,
    )

    print()
    print("PRIMARY")
    print(result.primary)
    print()
    print("SECONDARY")
    print(result.secondary)
    print()
    print("TOP-LEVEL DEBUG KEYS")
    print(result.debug_payload.keys())

    if result.channel_results:
        ch = result.channel_results[0]
        print()
        print("CHANNEL DEBUG KEYS")
        print(ch.debug_payload.keys())

        if "summary" in ch.debug_payload:
            print_channel_summary(ch.debug_payload["summary"])

        if "full" in ch.debug_payload:
            full = ch.debug_payload["full"]
            print()
            print("===== CHANNEL FULL (COUNTS) =====")
            print(f"top_note_candidates : {len(full.get('top_note_candidates', []))}")
            print(f"refined             : {len(full.get('refined', []))}")
            print(f"candidate_map       : {len(full.get('candidate_map', {}))}")
            print(f"band_rows           : {len(full.get('band_rows', []))}")


if __name__ == "__main__":
    main()