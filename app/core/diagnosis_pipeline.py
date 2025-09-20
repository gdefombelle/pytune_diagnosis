"""
diagnosis_pipeline.py
=====================
Pipeline principal pour l’analyse d’une note isolée (diagnostic PyTune).
"""

import numpy as np

from pytune_dsp.utils.yin import yin_track
from pytune_dsp.analysis.f0_analysis import stable_f0_detection
from pytune_dsp.analysis.partials import (
    compute_partials_fft_peaks,
    estimate_B,
)
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from app.models.schemas import NoteAnalysisResult


# ---------- Pipeline principal ----------

def analyze_note(
    note_name: str,
    expected_freq: float,
    signal: np.ndarray,
    sr: int,
) -> NoteAnalysisResult:
    """
    Analyse une note isolée : F0, validation, partiels, inharmonicité, spectre harmonique.
    """
    # Étape 1 : F0 stable via YIN restreint autour de la note attendue
    f0s = yin_track(signal, sr, expected_freq, semitones=0.5)
    stable_f0, mode_rate = stable_f0_detection(f0s)

    # Étape 2 : validation simple
    if stable_f0 < 20 or mode_rate < 0.4:
        print(f"❌ {note_name}: signal invalide (f0={stable_f0:.2f}Hz, mode_rate={mode_rate:.2f})")
        return NoteAnalysisResult(note_name=note_name, valid=False)

    deviation_cents = 1200 * np.log2(stable_f0 / expected_freq)
    if abs(deviation_cents) > 50:  # plus d’un demi-ton d’écart
        print(
            f"⚠️ {note_name}: hors tolérance (f0={stable_f0:.2f}Hz, "
            f"expected={expected_freq:.2f}Hz, dev={deviation_cents:.1f}cents)"
        )
        return NoteAnalysisResult(note_name=note_name, valid=False)

    print(
        f"✅ {note_name}: f0={stable_f0:.2f}Hz, expected={expected_freq:.2f}Hz, "
        f"dev={deviation_cents:.1f}cents, conf={mode_rate:.2f}"
    )

    # Étape 3 : calcul des partiels (FFT + interpolation) + inharmonicité
    nb_partials = 8
    harmonics, partials, inharmonicity = compute_partials_fft_peaks(
        signal=signal,
        sr=sr,
        f0_ref=stable_f0,
        nb_partials=nb_partials,
        search_width_cents=60.0,
        pad_factor=2,
    )

    partials_hz = [p[0] for p in partials]
    B = estimate_B(stable_f0, partials_hz) if partials_hz else 0.0

    # Logs console
    print("FFT partials")
    print(f"🎼 {note_name}: f0_ref={stable_f0:.2f}")
    if partials_hz:
        print(f"  f0_fft≈{partials_hz[0]:.2f} Hz (bin affiné)")
    print(f"  Harmonics (Hz) : {[round(h, 2) for h in harmonics]}")
    print(f"  Partials  (Hz) : {[round(f, 2) for f in partials_hz]}")
    print(f"  Inharm (cents) : {[round(c, 1) for c in inharmonicity]}")
    print(f"  B estimate     : {B:.3e}")

    # Étape 4 : empreinte spectrale et spectre harmonique
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum_norm = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    fingerprint = spectrum_norm[:512]

    harmonic_fp = harmonic_spectrum_fft(signal, sr, stable_f0, nb_harmonics=8)
    print("🎹 Harmonic fingerprint")
    for k, (f, amp) in enumerate(harmonic_fp, start=1):
        print(f"  H{k}: target={f:.2f} Hz, amplitude={amp:.3f}")

    return NoteAnalysisResult(
        note_name=note_name,
        valid=True,
        f0=stable_f0,
        confidence=mode_rate,
        deviation_cents=deviation_cents,
        expected_freq=expected_freq,
        harmonics=harmonics,
        partials=partials_hz,
        inharmonicity=inharmonicity,
        spectral_fingerprint=fingerprint,
        harmonic_spectrum=harmonic_fp,  # 👈 ajout ici
    )