# pytune_dsp/analysis/analyze.py
import numpy as np
from pathlib import Path
import csv
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from pytune_dsp.types.schemas import NoteAnalysisResult, GuessNoteResult
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import compute_inharmonicity_avg, estimate_B
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.preprocessing.trim import trim_signal
from pytune_dsp.utils.note_utils import freq_to_midi, freq_to_note

# v1 (librosa)
from pytune_dsp.analysis.guess_note import guess_note
# v2 (Essentia)
from pytune_dsp.analysis.guess_note_essentia import guess_note_essentia

DEBUG_CSV_PATH = Path("GUESS_NOTES_RESULTS.csv")


# ──────────────────────────────────────────────
# UTILITAIRES
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

def cents_between(f1: float, f2: float) -> float:
    return 1200.0 * np.log2(max(f1, 1e-12) / max(f2, 1e-12))


# ──────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE (double guess)
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
      1. Détection f₀ (librosa v0.7) ET (Essentia) en parallèle
      2. Sélection/fusion du meilleur f₀
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

    # ─────────────── STEP 1 : f₀ (deux méthodes en //) ───────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_lib = ex.submit(guess_note, signal, sr, True)
        fut_ess = ex.submit(guess_note_essentia, signal, sr, True)
        guess_lib = fut_lib.result()
        guess_ess = fut_ess.result()

    guesses = {"librosa": guess_lib, "essentia": guess_ess}
    if debug_csv:
        dump_guesses_to_csv(note_name, guesses)

    # Aucun résultat fiable
    if not (guess_lib and guess_lib.f0) and not (guess_ess and guess_ess.f0):
        print("⚠️ Aucun f₀ détecté — signal trop faible ou bruité.")
        return NoteAnalysisResult(
            note_name=note_name,
            valid=False,
            f0=None,
            confidence=0.0,
            deviation_cents=None,
            expected_freq=expected_freq,
        )

    # ─────────────── STEP 2 : sélection / fusion ───────────────
    best_guess: GuessNoteResult
    if (guess_lib and guess_lib.f0) and (guess_ess and guess_ess.f0):
        delta = abs(cents_between(guess_lib.f0, guess_ess.f0))
        # consensus fort → moyenne + boost de confiance
        if delta <= 15.0:
            f_mean = 0.5 * (guess_lib.f0 + guess_ess.f0)
            conf = min(1.0, max(guess_lib.confidence, guess_ess.confidence) + 0.15)
            best_guess = GuessNoteResult(
                midi=freq_to_midi(f_mean),
                f0=f_mean,
                confidence=conf,
                method=f"{guess_lib.method}+{guess_ess.method}",
                debug_log=(guess_lib.debug_log or []) + (guess_ess.debug_log or []),
                subresults={"librosa": asdict(guess_lib), "essentia": asdict(guess_ess)},
                envelope_band=guess_ess.envelope_band or guess_lib.envelope_band,
            )
        else:
            # sinon on prend la meilleure confiance (petit biais : en mid/high, préférer Essentia si confs proches)
            prefer_ess = (guess_ess.confidence + 0.05) >= guess_lib.confidence
            best_guess = guess_ess if prefer_ess else guess_lib
    else:
        best_guess = guess_ess if (guess_ess and guess_ess.f0) else guess_lib

    f0_est = best_guess.f0
    deviation_cents = 1200 * np.log2(f0_est / expected_freq) if expected_freq > 0 else None

    # stabilité minimale
    valid = (
        f0_est > 20.0
        and best_guess.confidence >= 0.30
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

    # ─────────────── RETURN ───────────────
    # on expose aussi les deux guesses pour inspection
    guesses["chosen"] = best_guess
    return NoteAnalysisResult(
        midi=int(round(midi_expected)),
        note_name=note_name,
        valid=valid,
        f0=f0_est,
        confidence=best_guess.confidence,
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
        guessed_note=asdict(best_guess),
        guesses={k: asdict(v) for k, v in guesses.items() if v},
        response=response_data,
    )