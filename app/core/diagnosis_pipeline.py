# pytune_dsp/analysis/analyze.py
import numpy as np
from pathlib import Path
import csv
from dataclasses import asdict
from pytune_dsp.types.schemas import NoteAnalysisResult, GuessNoteResult
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import compute_inharmonicity_avg, estimate_B
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.preprocessing.trim import trim_signal
from pytune_dsp.utils.note_utils import freq_to_midi, freq_to_note

# ⬇️ nouvelle fonction unique de détection
from pytune_dsp.analysis.guess_note import guess_note, guess_f0_fusion

DEBUG_CSV_PATH = Path("GUESS_NOTES_RESULTS.csv")


# ──────────────────────────────────────────────
# UTILITAIRE CSV DEBUG
# ──────────────────────────────────────────────
def dump_guesses_to_csv(note_name: str, guesses: dict):
    with DEBUG_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        row = [note_name]
        for k, g in guesses.items():
            if g and g.f0:
                row.append(f"{k}:{g.f0:.2f}Hz({g.confidence:.2f})")
            else:
                row.append(f"{k}:none")
        writer.writerow(row)


# ──────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE (nouvelle version)
# ──────────────────────────────────────────────
def analyze_note(
    note_name: str,
    expected_freq: float,
    signal: np.ndarray,
    sr: int,
    compute_inharm: bool = True,
    compute_response_data: bool = True,
    debug_csv: bool = False,
) -> NoteAnalysisResult:
    """
    Analyse complète d’une note capturée :
      1. Détection f₀ (guess_note fusionné : YIN + FFT + HPS + COMB)
      2. Validation de stabilité f₀
      3. Partiels & inharmonicité
      4. Empreinte spectrale
      5. Response optionnelle
    """
    if sr < 44100:
        raise ValueError(
            f"Sample rate too low ({sr} Hz). "
            "Please record at a minimum of 44.1 kHz for accurate PyTune analysis."
    )
    midi_expected = freq_to_midi(expected_freq)
    print(f"🎹 Expected {note_name} (MIDI {midi_expected}, {expected_freq:.2f} Hz)")

    # ─────────────── STEP 1 : f₀ estimation globale ───────────────
    guess = guess_note(signal, sr, debug=True)

    if not guess or not guess.f0:
        print("⚠️ Aucun f₀ détecté — signal trop faible ou bruité.")
        return NoteAnalysisResult(
            note_name=note_name,
            valid=False,
            f0=None,
            confidence=0.0,
            deviation_cents=None,
            expected_freq=expected_freq,
        )

    guesses = {"fusion": guess}
    if debug_csv:
        dump_guesses_to_csv(note_name, guesses)

    f0_est = guess.f0
    deviation_cents = (
        1200 * np.log2(f0_est / expected_freq)
        if expected_freq > 0 else None
    )

    # ─────────────── STEP 2 : validation de stabilité ───────────────
    # (ancien yin_track inutile : on garde f₀ stable venant du guess fusionné)
    yin_valid = (
        f0_est > 20.0
        and guess.confidence >= 0.3
        and abs(deviation_cents or 0) < 100
    )

    # ─────────────── STEP 3 : Partiels & inharmonicité ───────────────
    harmonics, partials, inharmonicity = compute_partials_fft_peaks(
        signal=signal,
        sr=sr,
        f0_ref=f0_est,
        nb_partials=8,
        search_width_cents=60.0,
        pad_factor=2,
    )

    partials_hz = [p[0] for p in partials]
    if compute_inharm:
        inharm_avg = compute_inharmonicity_avg(inharmonicity)
        B = estimate_B(f0_est, partials_hz) if partials_hz else None
    else:
        inharm_avg, B = None, None

    # ─────────────── STEP 4 : Spectral fingerprint ───────────────
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum_norm = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    fingerprint = spectrum_norm[:512]

    raw_fp, norm_fp = harmonic_spectrum_fft(signal, sr, f0_est, nb_harmonics=8)

    # ─────────────── STEP 5 : Response optionnelle ───────────────
    response_data = None
    if compute_response_data and len(signal) / sr >= 3.0:
        response_data = compute_response(signal, sr, f0_est)

    # ─────────────── RETURN NoteAnalysisResult ───────────────
    return NoteAnalysisResult(
        midi=int(round(midi_expected)),
        note_name=note_name,
        valid=yin_valid,
        f0=f0_est,
        confidence=guess.confidence,
        deviation_cents=deviation_cents,
        expected_freq=expected_freq,
        harmonics=harmonics,
        partials=partials_hz,
        inharmonicity=inharmonicity if compute_inharm else None,
        inharmonicity_avg=inharm_avg,
        B_estimate=B,
        spectral_fingerprint=fingerprint,
        harmonic_spectrum_raw=raw_fp,
        harmonic_spectrum_norm=norm_fp,

        # ✅ conversions dataclass → dict (pour Pydantic)
        guessed_note=asdict(guess) if guess else None,
        guesses={k: asdict(v) for k, v in guesses.items()},
        response=response_data,
    )