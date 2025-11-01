# pytune_diagnosis/app/core/diagnosis_pipeline.py
import numpy as np
from pathlib import Path
import csv
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from pytune_dsp.types.schemas import NoteAnalysisResult
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import compute_inharmonicity_avg, estimate_B
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.preprocessing.trim import trim_signal
from pytune_dsp.utils.note_utils import freq_to_midi
from pytune_dsp.utils.serialize import safe_asdict

# Pitch detection engines
from pytune_dsp.analysis.guess_note import guess_note            # v1 librosa
from pytune_dsp.analysis.guess_note_essentia import guess_note_essentia  # v2 essentia
from pytune_dsp.analysis.unison import analyze_unison             # new unison module
from pytune_dsp.analysis.f0_HP import f0_HP                       # new high-precision refinement

DEBUG_CSV_PATH = Path("GUESS_NOTES_RESULTS.csv")

# ──────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────
def dump_guesses_to_csv(note_name: str, guesses: dict):
    with DEBUG_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        row = [note_name]
        for k, g in guesses.items():
            if g and getattr(g, "f0", None):
                row.append(f"{k}:{g.f0:.2f}Hz({getattr(g, 'confidence', 0):.2f})")
            else:
                row.append(f"{k}:none")
        writer.writerow(row)


def cents_between(f1: float, f2: float) -> float:
    return 1200.0 * np.log2(max(f1, 1e-12) / max(f2, 1e-12))


# ──────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE (dual-engine + unison + f₀HP)
# ──────────────────────────────────────────────
def analyze_note(
    note_name: str,
    expected_freq: float,
    signal: np.ndarray,
    sr: int,
    compute_inharm: bool = True,
    compute_response_data: bool = True,
    debug_csv: bool = False,
    piano_type: str = "upright",
    era: str = "modern",
) -> NoteAnalysisResult:
    """
    Analyse complète d’une note capturée :
      1. Détection f₀ via 2 méthodes (librosa + essentia)
      2. Sélection/fusion du meilleur f₀
      3. Analyse d’unisson (nombre probable de cordes)
      4. Raffinement haute précision f₀_HP selon stabilité unisson
      5. Partiels & inharmonicité
      6. Empreinte spectrale + réponse temporelle
    """
    if sr < 44100:
        raise ValueError(
            f"Sample rate too low ({sr} Hz). "
            "Please record at a minimum of 44.1 kHz for accurate PyTune analysis."
        )

    midi_expected = freq_to_midi(expected_freq)
    print(f"🎹 Expected {note_name} (MIDI {midi_expected}, {expected_freq:.2f} Hz)")

    def timed(func, *args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        dur_ms = (time.perf_counter() - start) * 1000.0
        return res, dur_ms

    # ─────────────── STEP 1 : f₀ (dual-engine) ───────────────
    start_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_lib = ex.submit(timed, guess_note, signal, sr, True)
        fut_ess = ex.submit(timed, guess_note_essentia, signal, sr, True)
        (guess_lib, time_lib) = fut_lib.result()
        (guess_ess, time_ess) = fut_ess.result()
    total_time = (time.perf_counter() - start_parallel) * 1000.0

    print(f"⏱️ guess_note (librosa): {time_lib:.1f} ms")
    print(f"⏱️ guess_note_essentia: {time_ess:.1f} ms")
    print(f"⚙️ Temps total (parallèle): {total_time:.1f} ms")
    if total_time > 0:
        print(f"💡 Gain ≈ {(time_lib + time_ess) / total_time:.2f}× plus rapide qu’en séquentiel")

    guesses = {"librosa": guess_lib, "essentia": guess_ess}
    if debug_csv:
        dump_guesses_to_csv(note_name, guesses)

    # ─────────────── STEP 2 : sélection / fusion ───────────────
    valid_guesses = [
        g for g in guesses.values()
        if g and getattr(g, "f0", 0) > 0 and getattr(g, "confidence", 0) > 0
    ]
    if not valid_guesses:
        print("⚠️ Aucun f₀ détecté — signal trop faible ou bruité.")
        return NoteAnalysisResult(
            note_name=note_name,
            valid=False,
            f0=None,
            confidence=0.0,
            deviation_cents=None,
            expected_freq=expected_freq,
        )

    best_guess = max(valid_guesses, key=lambda g: getattr(g, "confidence", 0.0))
    for g in valid_guesses:
        if g is not best_guess and abs(cents_between(g.f0, best_guess.f0)) < 15.0:
            best_guess.confidence = min(1.0, best_guess.confidence + 0.1)

    f0_est = best_guess.f0
    deviation_cents = 1200 * np.log2(f0_est / expected_freq) if expected_freq > 0 else None
    valid = (
        f0_est > 20.0
        and best_guess.confidence >= 0.30
        and abs(deviation_cents or 0) < 100
    )

    print(f"🌟 Selected: {best_guess.method} | f0={f0_est:.2f} Hz | conf={best_guess.confidence:.2f}")

    # ─────────────── STEP 3 : analyse d’unisson ───────────────
    unison = analyze_unison(
        signal=signal,
        sr=sr,
        f0_est=f0_est,
        midi=int(round(midi_expected)),
        piano_type=piano_type,
        era=era,
    )

    print(f"🎛 Unison → posterior={unison.posterior}, "
          f"det={unison.detected_n_components}, "
          f"beat≈{unison.beat_hz_estimate:.2f}Hz, "
          f"severity={unison.severity:.2f}, "
          f"HP={unison.recommend_f0_hp} ({unison.confidence:.2f})")

    # ─────────────── STEP 4 : f₀ haute précision selon reco ───────────────
    f0_refined = f0_est
    if unison.recommend_f0_hp == "global":
        f0_refined = f0_HP(signal, sr, f0_seed=f0_est)
        print(f"🎯 f₀_HP (global) → {f0_refined:.4f} Hz")
    elif unison.recommend_f0_hp == "per-component" and len(unison.components_f0_band) >= 2:
        refined_components = []
        for comp in unison.components_f0_band:
            f_ref_i = f0_HP(signal, sr, f0_seed=comp.freq)
            refined_components.append((comp.freq, f_ref_i))
        # moyenne pondérée (approx) pour usage principal
        if refined_components:
            f0_refined = np.mean([r[1] for r in refined_components])
        print(f"🎯 f₀_HP (per-component) → {f0_refined:.4f} Hz (avg)")
    else:
        print("⚙️ f₀_HP skipped (unstable unison or low confidence)")

    # ─────────────── STEP 5 : Partiels & inharmonicité ───────────────
    harmonics, partials, inharmonicity = compute_partials_fft_peaks(
        signal=signal,
        sr=sr,
        f0_ref=f0_refined,
        nb_partials=8,
        search_width_cents=60.0,
        pad_factor=2,
    )

    partials_hz = [p[0] for p in partials]
    if compute_inharm:
        inharm_avg = compute_inharmonicity_avg(inharmonicity)
        B = estimate_B(f0_refined, partials_hz) if partials_hz else None
    else:
        inharm_avg, B = None, None

    # ─────────────── STEP 6 : Spectral fingerprint ───────────────
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum_norm = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    fingerprint = spectrum_norm[:512]
    raw_fp, norm_fp = harmonic_spectrum_fft(signal, sr, f0_refined, nb_harmonics=8)

    # ─────────────── STEP 7 : Response optionnelle ───────────────
    response_data = None
    if compute_response_data and len(signal) / sr >= 3.0:
        response_data = compute_response(signal, sr, f0_refined)

    # ─────────────── RETURN ───────────────
    guesses["chosen"] = best_guess
    return NoteAnalysisResult(
        midi=int(round(midi_expected)),
        note_name=note_name,
        valid=valid,
        f0=f0_refined,
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
        guessed_note=safe_asdict(best_guess),
        guesses={k: safe_asdict(v) for k, v in guesses.items() if v},
        response=response_data,
        unison=safe_asdict(unison),
    )