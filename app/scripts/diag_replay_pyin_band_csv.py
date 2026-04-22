import argparse
import csv
import math
import re
import wave
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import essentia.standard as es

from pytune_data.minio_client import (
    minio_client,
    PYTUNE_DIAGNOSIS_AUDIO_BUCKET,
)
from pytune_dsp.analysis.yin_backend_essentia_frames import yin_backend_essentia_frames
from pytune_dsp.utils.note_utils import midi_to_freq, freq_to_note
from pytune_dsp.analysis.hps_seq import (
    estimate_f0_hps_wrapper,
    estimate_f0_hps_multi_wrapper,
)


OBJECT_KEY_RE = re.compile(
    r"diagnosis/session_(?P<session_id>\d+|unknown)/notes/midi_(?P<midi>\d+)/(?P<filename>.+)\.wav$"
)

A0_HZ = 27.5
C8_HZ = 4186.01


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
    "pyin_band_f0",
    "pyin_band_note",
    "pyin_band_score",
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
    "candidate_top_notes",
]


@dataclass
class AudioSample:
    object_key: str
    session_id: Optional[int]
    expected_midi: int
    sample_rate: int
    signal: np.ndarray


@dataclass
class Band:
    index: int
    fmin: float
    fmax: float
    width_oct: float
    center_freq: float


def note_name_or_none(f: float | None) -> str | None:
    if f is None or f <= 0:
        return None
    return freq_to_note(f)


def cents_delta(f: Optional[float], ref: Optional[float]) -> Optional[float]:
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


def read_wav_from_minio(object_key: str) -> AudioSample:
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

    return AudioSample(
        object_key=object_key,
        session_id=session_id,
        expected_midi=expected_midi,
        sample_rate=sample_rate,
        signal=pcm,
    )


def build_log_bands(
    min_freq: float = A0_HZ,
    max_freq: float = C8_HZ,
    step_oct: float = 0.25,
) -> list[Band]:
    bands: list[Band] = []
    idx = 0
    f = min_freq
    while f < max_freq:
        f2 = min(max_freq, f * (2.0 ** step_oct))
        center = math.sqrt(f * f2)
        bands.append(
            Band(
                index=idx,
                fmin=float(f),
                fmax=float(f2),
                width_oct=float(step_oct),
                center_freq=float(center),
            )
        )
        idx += 1
        f = f2
    return bands


def fft_band_f0(
    signal: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
) -> tuple[Optional[float], float]:
    y = np.asarray(signal, dtype=np.float32)
    if len(y) < 256:
        return None, 0.0

    y = y * np.hanning(len(y))
    nfft = max(8192, 1 << int(np.ceil(np.log2(len(y)))))
    spec = np.fft.rfft(y, n=nfft)
    mag = np.abs(spec).astype(np.float64)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    lo = int(np.searchsorted(freqs, fmin, side="left"))
    hi = int(np.searchsorted(freqs, fmax, side="right"))
    if hi <= lo:
        return None, 0.0

    seg_freqs = freqs[lo:hi]
    seg_mag = mag[lo:hi].copy()
    if seg_mag.size == 0:
        return None, 0.0

    seg_mag *= 1.0 / np.sqrt(np.maximum(seg_freqs, 1e-6))

    idx_local = int(np.argmax(seg_mag))
    peak_mag = float(seg_mag[idx_local])
    if peak_mag <= 0:
        return None, 0.0

    idx = lo + idx_local
    f0 = float(freqs[idx])

    baseline = float(np.median(seg_mag) + 1e-12)
    conf = float(np.clip((peak_mag / baseline) / 20.0, 0.0, 1.0))
    return f0, conf


def centered_bounds(center_freq: float, window_cents: float) -> tuple[float, float]:
    ratio = 2.0 ** (window_cents / 1200.0)
    return max(20.0, center_freq / ratio), center_freq * ratio


def bandpass_signal_fft(
    signal: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    pad_factor: int = 2,
) -> np.ndarray:
    y = np.asarray(signal, dtype=np.float32)
    if y.size < 256:
        return y.copy()

    nfft = max(8192, pad_factor * (1 << int(np.ceil(np.log2(len(y))))))
    spec = np.fft.rfft(y, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    mask = (freqs >= fmin) & (freqs <= fmax)
    spec_filtered = np.zeros_like(spec)
    spec_filtered[mask] = spec[mask]

    y_filt = np.fft.irfft(spec_filtered, n=nfft)[: len(y)]
    y_filt = np.asarray(y_filt, dtype=np.float32)

    peak = float(np.max(np.abs(y_filt))) if y_filt.size else 0.0
    if peak > 1e-12:
        y_filt = y_filt / peak

    return y_filt


def choose_pyin_frame_size(center_freq: float, signal_len: int, sr: int) -> int:
    if center_freq < 80.0:
        frame_size = 8192
    elif center_freq < 300.0:
        frame_size = 4096
    elif center_freq < 1200.0:
        frame_size = 2048
    else:
        frame_size = 1024

    max_allowed = 1 << int(np.floor(np.log2(max(signal_len, 512))))
    frame_size = min(frame_size, max_allowed)
    frame_size = max(frame_size, 512)
    return int(frame_size)


def compute_pyin_band(
    signal: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    center_freq: float,
) -> tuple[Optional[float], Optional[str], Optional[float]]:
    y_band = bandpass_signal_fft(signal, sr, fmin, fmax)
    if len(y_band) < 512:
        return None, None, None

    frame_size = choose_pyin_frame_size(center_freq, len(y_band), sr)
    hop_size = max(128, frame_size // 8)

    try:
        algo = es.PitchYinProbabilistic(
            frameSize=int(frame_size),
            hopSize=int(hop_size),
            sampleRate=float(sr),
            outputUnvoiced="negative",
            preciseTime=False,
        )
        pitch, voiced = algo(y_band)
    except Exception:
        return None, None, None

    valid_pitch = []
    valid_voiced = []

    for p, v in zip(pitch, voiced):
        try:
            pf = float(p)
            vf = float(v)
        except Exception:
            continue
        if pf > 0 and fmin <= pf <= fmax:
            valid_pitch.append(pf)
            valid_voiced.append(vf)

    if not valid_pitch:
        return None, None, None

    ps = np.asarray(valid_pitch, dtype=np.float64)
    vs = np.asarray(valid_voiced, dtype=np.float64)

    pyin_f0 = float(np.median(ps))
    pyin_score = float(np.mean(vs)) if len(vs) else None
    pyin_note = note_name_or_none(pyin_f0)

    return pyin_f0, pyin_note, pyin_score


def select_best_fft_band_candidate(
    signal: np.ndarray,
    sr: int,
    band_rows: list[dict],
    top_k: int = 3,
    refine_window_cents: float = 100.0,
) -> tuple[Optional[str], Optional[float], float]:
    candidates = [
        r for r in band_rows
        if r.get("fft_band_note") is not None
        and r.get("fft_band_f0") is not None
    ]
    if not candidates:
        return None, None, 0.0

    best_per_note: dict[str, dict] = {}
    for r in candidates:
        note = str(r["fft_band_note"])
        prev = best_per_note.get(note)
        if prev is None or float(r.get("fft_band_score") or 0.0) > float(prev.get("fft_band_score") or 0.0):
            best_per_note[note] = r

    top_candidates = sorted(
        best_per_note.values(),
        key=lambda r: float(r.get("fft_band_score") or 0.0),
        reverse=True,
    )[:top_k]

    best_note = None
    best_f0 = None
    best_score = -1.0

    for r in top_candidates:
        candidate_f0 = float(r["fft_band_f0"])
        candidate_note = str(r["fft_band_note"])
        base_band_score = float(r.get("fft_band_score") or 0.0)

        fmin, fmax = centered_bounds(candidate_f0, refine_window_cents)
        local_f0, local_score = fft_band_f0(
            signal=signal,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
        )
        local_note = note_name_or_none(local_f0)

        same_note_bonus = 1.0 if local_note == candidate_note else 0.0

        discriminant = (
            1.50 * float(local_score or 0.0)
            + 0.20 * float(base_band_score or 0.0)
            + 0.50 * same_note_bonus
        )

        if discriminant > best_score:
            best_score = discriminant
            best_note = candidate_note
            best_f0 = candidate_f0

    return best_note, best_f0, float(best_score)


def scan_bands(
    signal: np.ndarray,
    sr: int,
    yin_bands: list[Band],
    fft_bands: list[Band],
    direction: str,
    stop_midi: Optional[float] = None,
) -> list[dict]:
    if direction == "low_to_high":
        iter_yin_bands = yin_bands
    elif direction == "high_to_low":
        iter_yin_bands = list(reversed(yin_bands))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    rows: list[dict] = []

    for scan_rank, band in enumerate(iter_yin_bands):
        center_midi = 69 + 12 * math.log2(max(band.center_freq, 1e-9) / 440.0)

        if stop_midi is not None:
            if direction == "low_to_high" and center_midi > stop_midi:
                break
            if direction == "high_to_low" and center_midi < stop_midi:
                break

        f0, score = yin_backend_essentia_frames(
            signal=signal,
            sr=sr,
            fmin=band.fmin,
            fmax=band.fmax,
        )

        if f0 is None:
            continue

        try:
            f0 = float(f0)
            score = float(score)
        except Exception:
            continue

        fft_band_match = None
        for fb in fft_bands:
            if fb.fmin <= f0 <= fb.fmax:
                fft_band_match = fb
                break

        if fft_band_match is None:
            fft_band_match = min(
                fft_bands,
                key=lambda fb: abs(math.log2(max(f0, 1e-9) / fb.center_freq))
            )

        fft_band_f0_value, fft_band_score = fft_band_f0(
            signal=signal,
            sr=sr,
            fmin=fft_band_match.fmin,
            fmax=fft_band_match.fmax,
        )

        pyin_band_f0, pyin_band_note, pyin_band_score = compute_pyin_band(
            signal=signal,
            sr=sr,
            fmin=band.fmin,
            fmax=band.fmax,
            center_freq=band.center_freq,
        )

        rows.append(
            {
                "scan_direction": direction,
                "scan_rank": scan_rank,
                "band_index": band.index,
                "band_center_freq": band.center_freq,
                "band_fmin": band.fmin,
                "band_fmax": band.fmax,
                "candidate_f0": f0,
                "candidate_note": note_name_or_none(f0),
                "pitchyinfft_score": score,
                "candidate_in_band": band.fmin <= f0 <= band.fmax,
                "fft_band_f0": fft_band_f0_value,
                "fft_band_note": note_name_or_none(fft_band_f0_value),
                "fft_band_score": fft_band_score,
                "pyin_band_f0": pyin_band_f0,
                "pyin_band_note": pyin_band_note,
                "pyin_band_score": pyin_band_score,
            }
        )

    return rows


def refine_candidate(
    signal: np.ndarray,
    sr: int,
    candidate_f0: float,
    candidate_note: str,
    window_cents: float,
) -> dict:
    fmin, fmax = centered_bounds(candidate_f0, window_cents)
    f0, score = yin_backend_essentia_frames(
        signal=signal,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
    )
    return {
        "refined_from_note": candidate_note,
        "refined_from_f0": candidate_f0,
        "refined_f0": f0,
        "refined_note": note_name_or_none(f0),
        "refined_score": score,
    }


def compute_hps_non_informed(
    signal: np.ndarray,
    sr: int,
) -> dict:
    hps = estimate_f0_hps_wrapper(
        signal,
        sr,
        tessitura="auto",
        expected_f0=None,
        expected_weight=0.0,
    )

    hps_multi = estimate_f0_hps_multi_wrapper(
        signal,
        sr,
        top_k=5,
        tessitura="auto",
        expected_f0=None,
    )

    hps_f0 = None
    hps_note = None
    hps_score = None
    hps_multi_note = None

    if isinstance(hps, dict):
        f0 = hps.get("f0")
        q = hps.get("quality")
        try:
            if f0 and float(f0) > 0:
                hps_f0 = float(f0)
                hps_note = note_name_or_none(hps_f0)
        except Exception:
            pass
        try:
            if q is not None:
                hps_score = float(q)
        except Exception:
            pass

    if isinstance(hps_multi, dict):
        cands = hps_multi.get("candidates", [])
        if cands:
            best = cands[0]
            try:
                f0m = float(best.get("f0", 0.0))
                if f0m > 0:
                    hps_multi_note = note_name_or_none(f0m)
            except Exception:
                pass

    return {
        "hps_f0": hps_f0,
        "hps_note": hps_note,
        "hps_score": hps_score,
        "hps_multi_note": hps_multi_note,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="CSV YIN scan + non-informed fft_band + HPS global + pYIN by band."
    )
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="tmp/diag_yin_simple_csv")
    parser.add_argument("--band-mode", type=str, default="octave", choices=["octave", "half_octave", "quarter_octave"])
    parser.add_argument("--stop-midi", type=float, default=64.0)
    parser.add_argument("--score-threshold", type=float, default=0.80)
    parser.add_argument("--top-notes", type=int, default=4)
    parser.add_argument("--refine-window-cents", type=float, default=100.0)
    args = parser.parse_args()

    if args.band_mode == "octave":
        yin_step_oct = 1.0
    elif args.band_mode == "half_octave":
        yin_step_oct = 0.5
    else:
        yin_step_oct = 0.25

    yin_bands = build_log_bands(step_oct=yin_step_oct)
    fft_bands = build_log_bands(step_oct=1.0 / 3.0)

    keys = list_session_audio_keys(args.session_id)
    if not keys:
        raise RuntimeError(f"No diagnosis audio files found for session {args.session_id}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"diag_session_{args.session_id}_yin_simple_csv_{args.band_mode}_{stamp}.csv"

    rows_out: list[dict] = []

    print(f"Found {len(keys)} WAV samples in MinIO for session {args.session_id}")

    for idx, key in enumerate(keys, start=1):
        sample = read_wav_from_minio(key)
        played_freq = midi_to_freq(sample.expected_midi)
        played_note = freq_to_note(played_freq)

        band_rows = scan_bands(
            sample.signal,
            sample.sample_rate,
            yin_bands,
            fft_bands,
            "low_to_high",
            args.stop_midi,
        )
        band_rows += scan_bands(
            sample.signal,
            sample.sample_rate,
            yin_bands,
            fft_bands,
            "high_to_low",
            args.stop_midi,
        )

        fft_band_selected, fft_band_selected_f0, fft_band_selected_score = select_best_fft_band_candidate(
            signal=sample.signal,
            sr=sample.sample_rate,
            band_rows=band_rows,
            top_k=3,
            refine_window_cents=args.refine_window_cents,
        )

        hps_info = compute_hps_non_informed(
            signal=sample.signal,
            sr=sample.sample_rate,
        )

        filtered_candidates = [
            r for r in band_rows
            if r["candidate_note"] is not None
            and r["candidate_in_band"] is True
            and float(r["pitchyinfft_score"]) >= args.score_threshold
        ]

        best_per_note: dict[str, dict] = {}
        for r in filtered_candidates:
            note = str(r["candidate_note"])
            prev = best_per_note.get(note)
            if prev is None or float(r["pitchyinfft_score"]) > float(prev["pitchyinfft_score"]):
                best_per_note[note] = r

        top_note_candidates = sorted(
            best_per_note.values(),
            key=lambda r: float(r["pitchyinfft_score"]),
            reverse=True,
        )[: args.top_notes]

        refined = []
        for r in top_note_candidates:
            ref = refine_candidate(
                sample.signal,
                sample.sample_rate,
                float(r["candidate_f0"]),
                str(r["candidate_note"]),
                args.refine_window_cents,
            )
            ref["refined_delta_cents"] = cents_delta(ref["refined_f0"], played_freq)

            refined.append(
                {
                    **r,
                    **ref,
                }
            )

        candidate_top_notes_list = [str(r["candidate_note"]) for r in top_note_candidates]
        candidate_top_notes = ", ".join(candidate_top_notes_list)

        chosen = None
        chosen_method = None

        if refined:
            refined.sort(
                key=lambda r: (
                    float(r["refined_score"] or 0.0),
                    float(r["pitchyinfft_score"] or 0.0),
                ),
                reverse=True,
            )
            chosen = refined[0]
            chosen_method = "best_refined_score"

        if fft_band_selected is not None and fft_band_selected in candidate_top_notes_list:
            chosen = {
                "refined_note": fft_band_selected,
                "refined_f0": fft_band_selected_f0,
                "refined_score": fft_band_selected_score,
            }
            chosen_method = "fft_band_selected_in_yin_candidates"

        band_rows = sorted(
            band_rows,
            key=lambda r: float(r.get("fft_band_score") or 0.0),
            reverse=True,
        )

        for r in band_rows:
            matched_ref = None
            note = r["candidate_note"]
            if note is not None:
                for ref in refined:
                    if ref["candidate_note"] == note:
                        matched_ref = ref
                        break

            rows_out.append(
                {
                    "row_type": "band_row",
                    "played_note": played_note,
                    "played_midi": sample.expected_midi,
                    "played_freq": played_freq,
                    "band_index": r["band_index"],
                    "band_center_freq": r["band_center_freq"],
                    "band_fmin": r["band_fmin"],
                    "band_fmax": r["band_fmax"],
                    "scan_direction": r["scan_direction"],
                    "candidate_f0": r["candidate_f0"],
                    "candidate_note": r["candidate_note"],
                    "pitchyinfft_score": r["pitchyinfft_score"],
                    "candidate_in_band": r["candidate_in_band"],
                    "delta_cents_vs_played": cents_delta(r["candidate_f0"], played_freq),
                    "fft_band_f0": r.get("fft_band_f0"),
                    "fft_band_note": r.get("fft_band_note"),
                    "fft_band_score": r.get("fft_band_score"),
                    "fft_band_selected": fft_band_selected,
                    "hps_f0": hps_info["hps_f0"],
                    "hps_note": hps_info["hps_note"],
                    "hps_score": hps_info["hps_score"],
                    "hps_multi_note": hps_info["hps_multi_note"],
                    "pyin_band_f0": r.get("pyin_band_f0"),
                    "pyin_band_note": r.get("pyin_band_note"),
                    "pyin_band_score": r.get("pyin_band_score"),
                    "refine_selected": matched_ref is not None,
                    "refined_from_note": matched_ref["refined_from_note"] if matched_ref else None,
                    "refined_from_f0": matched_ref["refined_from_f0"] if matched_ref else None,
                    "refined_f0": matched_ref["refined_f0"] if matched_ref else None,
                    "refined_note": matched_ref["refined_note"] if matched_ref else None,
                    "refined_score": matched_ref["refined_score"] if matched_ref else None,
                    "refined_delta_cents": matched_ref["refined_delta_cents"] if matched_ref else None,
                    "chosen_note": chosen["refined_note"] if chosen else None,
                    "chosen_f0": chosen["refined_f0"] if chosen else None,
                    "chosen_score": chosen["refined_score"] if chosen else None,
                    "chosen_method": chosen_method,
                    "candidate_top_notes": candidate_top_notes,
                }
            )

        rows_out.append(
            {
                "row_type": "summary",
                "played_note": played_note,
                "played_midi": sample.expected_midi,
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
                "hps_f0": hps_info["hps_f0"],
                "hps_note": hps_info["hps_note"],
                "hps_score": hps_info["hps_score"],
                "hps_multi_note": hps_info["hps_multi_note"],
                "pyin_band_f0": None,
                "pyin_band_note": None,
                "pyin_band_score": None,
                "refine_selected": None,
                "refined_from_note": None,
                "refined_from_f0": None,
                "refined_f0": None,
                "refined_note": None,
                "refined_score": None,
                "refined_delta_cents": None,
                "chosen_note": chosen["refined_note"] if chosen else None,
                "chosen_f0": chosen["refined_f0"] if chosen else None,
                "chosen_score": chosen["refined_score"] if chosen else None,
                "chosen_method": chosen_method if chosen else "no_candidate",
                "candidate_top_notes": candidate_top_notes,
            }
        )

        print(
            f"[{idx}/{len(keys)}] played={played_note} "
            f"top_notes={candidate_top_notes} "
            f"fft_band_selected={fft_band_selected} "
            f"hps={hps_info['hps_note']} "
            f"hps_multi={hps_info['hps_multi_note']} "
            f"chosen={chosen['refined_note'] if chosen else None} "
            f"method={chosen_method}"
        )

    write_csv(rows_out, csv_path)
    print()
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()