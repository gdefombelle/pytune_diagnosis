import argparse
import csv
import re
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

from pytune_data.minio_client import (
    PYTUNE_DIAGNOSIS_AUDIO_BUCKET,
    minio_client,
)
from pytune_dsp.analysis.detect_f0_seed import detect_f0_seed
from pytune_dsp.types.dataclasses import AudioChannelInput
from pytune_dsp.utils.note_utils import midi_to_freq, freq_to_note


OBJECT_KEY_RE = re.compile(
    r"diagnosis/session_(?P<session_id>\d+|unknown)/notes/midi_(?P<midi>\d+)/(?P<filename>.+)\.wav$"
)

CSV_COLUMNS = [
    "row_type",
    "played_note",
    "played_midi",
    "played_freq",
    "band_index",
    "band_center_freq",
    "band_fmin",
    "band_fmax",
    "scan_direction",
    "candidate_f0",
    "candidate_note",
    "pitchyinfft_score",
    "candidate_in_band",
    "delta_cents_vs_played",
    "fft_band_f0",
    "fft_band_note",
    "fft_band_score",
    "fft_band_selected",
    "hps_f0",
    "hps_note",
    "hps_score",
    "hps_multi_note",
    "candidate_fft_local_f0",
    "candidate_fft_local_note",
    "candidate_fft_local_score",
    "candidate_hps_local_f0",
    "candidate_hps_local_note",
    "candidate_hps_local_score",
    "candidate_hps_band_note",
    "candidate_hps_agreement",
    "candidate_final_score",
    "refine_selected",
    "refined_from_note",
    "refined_from_f0",
    "refined_f0",
    "refined_note",
    "refined_score",
    "refined_delta_cents",
    "chosen_note",
    "chosen_f0",
    "chosen_score",
    "chosen_method",
    "chosen_note_2",
    "chosen_f0_2",
    "chosen_score_2",
    "chosen_method_2",
    "chosen_note_3",
    "chosen_f0_3",
    "chosen_score_3",
    "chosen_method_3",
    "chosen_note_4",
    "chosen_f0_4",
    "chosen_score_4",
    "chosen_method_4",
    "candidate_top_notes",
]


def cents_delta(f: Optional[float], ref: Optional[float]) -> Optional[float]:
    import math

    if not f or not ref or f <= 0 or ref <= 0:
        return None
    return 1200.0 * math.log2(f / ref)


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

    def midi_sort_key(k: str) -> tuple[int, str]:
        try:
            _, midi = parse_object_key(k)
            return (midi, k)
        except Exception:
            return (9999, k)

    return sorted(keys, key=midi_sort_key)


def read_wav_from_minio(object_key: str) -> tuple[np.ndarray, int, Optional[int], int]:
    session_id, expected_midi = parse_object_key(object_key)

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

    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return pcm, sample_rate, session_id, expected_midi


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Replay validation using detect_f0_seed() canonical API."
    )
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="tmp/diag_detect_f0_seed_csv")
    parser.add_argument("--band-mode", type=str, default="octave", choices=["octave", "half_octave", "quarter_octave"])
    parser.add_argument("--stop-midi", type=float, default=64.0)
    parser.add_argument("--score-threshold", type=float, default=0.80)
    parser.add_argument("--top-notes", type=int, default=4)
    parser.add_argument("--refine-window-cents", type=float, default=100.0)
    parser.add_argument("--octave-min-gain", type=float, default=0.35)
    parser.add_argument("--chosen4-min-gain", type=float, default=0.20)
    parser.add_argument("--chosen4-hps-threshold", type=float, default=0.70)
    parser.add_argument("--min-freq", type=float, default=27.5)
    parser.add_argument("--max-freq", type=float, default=4186.01)
    args = parser.parse_args()

    keys = list_session_audio_keys(args.session_id)
    if not keys:
        raise RuntimeError(f"No diagnosis audio files found for session {args.session_id}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"diag_session_{args.session_id}_detect_f0_seed_{args.band_mode}_{stamp}.csv"

    rows_out: list[dict] = []

    print(f"Found {len(keys)} WAV samples in MinIO for session {args.session_id}")

    for idx, key in enumerate(keys, start=1):
        signal, sample_rate, _, expected_midi = read_wav_from_minio(key)
        played_freq = midi_to_freq(expected_midi)
        played_note = freq_to_note(played_freq)

        result = detect_f0_seed(
            channels=[
                AudioChannelInput(
                    signal=signal,
                    sample_rate=sample_rate,
                    channel_id="mic1",
                    mic_name="mic1",
                )
            ],
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            max_channels_to_analyze=1,
            band_mode=args.band_mode,
            stop_midi=args.stop_midi,
            score_threshold=args.score_threshold,
            top_notes=args.top_notes,
            refine_window_cents=args.refine_window_cents,
            octave_min_gain=args.octave_min_gain,
            chosen4_min_gain=args.chosen4_min_gain,
            chosen4_hps_threshold=args.chosen4_hps_threshold,
            debug=True,
        )

        channel_result = result.channel_results[0] if result.channel_results else None
        debug_payload = channel_result.debug_payload if channel_result else {}

        band_rows = debug_payload.get("band_rows", [])
        hps_info = debug_payload.get("hps_info", {})
        candidate_map = debug_payload.get("candidate_map", {})
        refined = debug_payload.get("refined", [])
        candidate_top_notes = debug_payload.get("candidate_top_notes", "")

        chosen = debug_payload.get("chosen")
        chosen_method = debug_payload.get("chosen_method")

        chosen2 = debug_payload.get("chosen2")
        chosen_method_2 = debug_payload.get("chosen_method_2")

        chosen3 = debug_payload.get("chosen3")
        chosen_method_3 = debug_payload.get("chosen_method_3")

        chosen4 = debug_payload.get("chosen4")
        chosen_method_4 = debug_payload.get("chosen_method_4")

        fft_band_selected = debug_payload.get("fft_band_selected")

        band_rows = sorted(
            band_rows,
            key=lambda r: float(r.get("fft_band_score") or 0.0),
            reverse=True,
        )

        for r in band_rows:
            matched_ref = None
            note = r.get("candidate_note")
            if note is not None:
                for ref in refined:
                    if ref.get("candidate_note") == note:
                        matched_ref = ref
                        break

            candidate_extra = candidate_map.get(str(note), {})

            rows_out.append(
                {
                    "row_type": "band_row",
                    "played_note": played_note,
                    "played_midi": expected_midi,
                    "played_freq": played_freq,
                    "band_index": r.get("band_index"),
                    "band_center_freq": r.get("band_center_freq"),
                    "band_fmin": r.get("band_fmin"),
                    "band_fmax": r.get("band_fmax"),
                    "scan_direction": r.get("scan_direction"),
                    "candidate_f0": r.get("candidate_f0"),
                    "candidate_note": r.get("candidate_note"),
                    "pitchyinfft_score": r.get("pitchyinfft_score"),
                    "candidate_in_band": r.get("candidate_in_band"),
                    "delta_cents_vs_played": cents_delta(r.get("candidate_f0"), played_freq),
                    "fft_band_f0": r.get("fft_band_f0"),
                    "fft_band_note": r.get("fft_band_note"),
                    "fft_band_score": r.get("fft_band_score"),
                    "fft_band_selected": fft_band_selected,
                    "hps_f0": hps_info.get("hps_f0"),
                    "hps_note": hps_info.get("hps_note"),
                    "hps_score": hps_info.get("hps_score"),
                    "hps_multi_note": hps_info.get("hps_multi_note"),
                    "candidate_fft_local_f0": candidate_extra.get("candidate_fft_local_f0"),
                    "candidate_fft_local_note": candidate_extra.get("candidate_fft_local_note"),
                    "candidate_fft_local_score": candidate_extra.get("candidate_fft_local_score"),
                    "candidate_hps_local_f0": candidate_extra.get("candidate_hps_local_f0"),
                    "candidate_hps_local_note": candidate_extra.get("candidate_hps_local_note"),
                    "candidate_hps_local_score": candidate_extra.get("candidate_hps_local_score"),
                    "candidate_hps_band_note": candidate_extra.get("candidate_hps_band_note"),
                    "candidate_hps_agreement": candidate_extra.get("candidate_hps_agreement"),
                    "candidate_final_score": candidate_extra.get("candidate_final_score"),
                    "refine_selected": matched_ref is not None,
                    "refined_from_note": matched_ref.get("refined_from_note") if matched_ref else None,
                    "refined_from_f0": matched_ref.get("refined_from_f0") if matched_ref else None,
                    "refined_f0": matched_ref.get("refined_f0") if matched_ref else None,
                    "refined_note": matched_ref.get("refined_note") if matched_ref else None,
                    "refined_score": matched_ref.get("refined_score") if matched_ref else None,
                    "refined_delta_cents": matched_ref.get("refined_delta_cents") if matched_ref else None,
                    "chosen_note": chosen.get("refined_note") if chosen else None,
                    "chosen_f0": chosen.get("refined_f0") if chosen else None,
                    "chosen_score": chosen.get("refined_score") if chosen else None,
                    "chosen_method": chosen_method,
                    "chosen_note_2": chosen2.get("refined_note") if chosen2 else None,
                    "chosen_f0_2": chosen2.get("refined_f0") if chosen2 else None,
                    "chosen_score_2": chosen2.get("refined_score") if chosen2 else None,
                    "chosen_method_2": chosen_method_2 if chosen2 else "no_candidate",
                    "chosen_note_3": chosen3.get("refined_note") if chosen3 else None,
                    "chosen_f0_3": chosen3.get("refined_f0") if chosen3 else None,
                    "chosen_score_3": chosen3.get("refined_score") if chosen3 else None,
                    "chosen_method_3": chosen_method_3 if chosen3 else "no_candidate",
                    "chosen_note_4": chosen4.get("refined_note") if chosen4 else None,
                    "chosen_f0_4": chosen4.get("refined_f0") if chosen4 else None,
                    "chosen_score_4": chosen4.get("refined_score") if chosen4 else None,
                    "chosen_method_4": chosen_method_4 if chosen4 else "no_candidate",
                    "candidate_top_notes": candidate_top_notes,
                }
            )

        rows_out.append(
            {
                "row_type": "summary",
                "played_note": played_note,
                "played_midi": expected_midi,
                "played_freq": played_freq,
                "band_index": None,
                "band_center_freq": None,
                "band_fmin": None,
                "band_fmax": None,
                "scan_direction": None,
                "candidate_f0": None,
                "candidate_note": None,
                "pitchyinfft_score": None,
                "candidate_in_band": None,
                "delta_cents_vs_played": None,
                "fft_band_f0": None,
                "fft_band_note": None,
                "fft_band_score": None,
                "fft_band_selected": fft_band_selected,
                "hps_f0": hps_info.get("hps_f0"),
                "hps_note": hps_info.get("hps_note"),
                "hps_score": hps_info.get("hps_score"),
                "hps_multi_note": hps_info.get("hps_multi_note"),
                "candidate_fft_local_f0": None,
                "candidate_fft_local_note": None,
                "candidate_fft_local_score": None,
                "candidate_hps_local_f0": None,
                "candidate_hps_local_note": None,
                "candidate_hps_local_score": None,
                "candidate_hps_band_note": None,
                "candidate_hps_agreement": None,
                "candidate_final_score": None,
                "refine_selected": None,
                "refined_from_note": None,
                "refined_from_f0": None,
                "refined_f0": None,
                "refined_note": None,
                "refined_score": None,
                "refined_delta_cents": None,
                "chosen_note": chosen.get("refined_note") if chosen else None,
                "chosen_f0": chosen.get("refined_f0") if chosen else None,
                "chosen_score": chosen.get("refined_score") if chosen else None,
                "chosen_method": chosen_method if chosen else "no_candidate",
                "chosen_note_2": chosen2.get("refined_note") if chosen2 else None,
                "chosen_f0_2": chosen2.get("refined_f0") if chosen2 else None,
                "chosen_score_2": chosen2.get("refined_score") if chosen2 else None,
                "chosen_method_2": chosen_method_2 if chosen2 else "no_candidate",
                "chosen_note_3": chosen3.get("refined_note") if chosen3 else None,
                "chosen_f0_3": chosen3.get("refined_f0") if chosen3 else None,
                "chosen_score_3": chosen3.get("refined_score") if chosen3 else None,
                "chosen_method_3": chosen_method_3 if chosen3 else "no_candidate",
                "chosen_note_4": chosen4.get("refined_note") if chosen4 else None,
                "chosen_f0_4": chosen4.get("refined_f0") if chosen4 else None,
                "chosen_score_4": chosen4.get("refined_score") if chosen4 else None,
                "chosen_method_4": chosen_method_4 if chosen4 else "no_candidate",
                "candidate_top_notes": candidate_top_notes,
            }
        )

        print(
            f"[{idx}/{len(keys)}] played={played_note} "
            f"chosen={chosen.get('refined_note') if chosen else None} "
            f"chosen2={chosen2.get('refined_note') if chosen2 else None} "
            f"chosen3={chosen3.get('refined_note') if chosen3 else None} "
            f"chosen4={chosen4.get('refined_note') if chosen4 else None}"
        )

    write_csv(rows_out, csv_path)
    print()
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()