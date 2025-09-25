# pytune_dsp/analysis/analyze.py
from app.models.schemas import NoteAnalysisResult
import numpy as np

from pytune_dsp.utils.yin import yin_track
from pytune_dsp.analysis.f0_analysis import stable_f0_detection
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import (
    compute_inharmonicity_avg,
    estimate_B,
)
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.preprocessing.trim import trim_signal
from pytune_dsp.analysis.guess_note import guess_f0_fft, guess_f0_pattern, guess_f0_fusion


def analyze_note(
    note_name: str,
    expected_freq: float,
    signal: np.ndarray,
    sr: int,
    compute_inharm: bool = True,
):
    # Pré-traitement
    signal = trim_signal(signal, sr)

    # Étape 1 : YIN tracking restreint autour de expected_freq
    f0s = yin_track(signal, sr, expected_freq, semitones=0.5)
    stable_f0, mode_rate = stable_f0_detection(f0s)

    # Étape 2 : Validation YIN
    yin_valid = True
    if stable_f0 < 20 or mode_rate < 0.4:
        yin_valid = False

    deviation_cents = 1200 * np.log2(stable_f0 / expected_freq) if stable_f0 > 0 else None
    if deviation_cents is not None and abs(deviation_cents) > 50:
        yin_valid = False

    # Étape 3 : partiels et inharmonicité
    nb_partials = 8
    harmonics, partials, inharmonicity = compute_partials_fft_peaks(
        signal=signal,
        sr=sr,
        f0_ref=stable_f0 if stable_f0 > 0 else expected_freq,
        nb_partials=nb_partials,
        search_width_cents=60.0,
        pad_factor=2,
    )

    partials_hz = [p[0] for p in partials]

    if compute_inharm:
        inharm_avg = compute_inharmonicity_avg(inharmonicity)
        B = estimate_B(stable_f0, partials_hz) if partials_hz else None
    else:
        inharmonicity = []
        inharm_avg = None
        B = None

    # Étape 4 : empreinte spectrale
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum_norm = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    fingerprint = spectrum_norm[:512]

    raw_fp, norm_fp = harmonic_spectrum_fft(
        signal,
        sr,
        stable_f0 if stable_f0 > 0 else expected_freq,
        nb_harmonics=8,
    )

    # --- Étape 5a : Guess basé sur expected_freq (FFT harmonics)
    f0_guess_fft, conf_guess_fft = guess_f0_fft(signal, sr, expected_freq)
    if f0_guess_fft:
        dev_guess_fft = 1200 * np.log2(f0_guess_fft / expected_freq)
        print(
            f"🔍 Guess FFT: f0={f0_guess_fft:.2f}Hz "
            f"(dev={dev_guess_fft:+.1f}¢, conf={conf_guess_fft:.2f})"
        )
        if stable_f0 > 0:
            diff_cents = 1200 * np.log2(f0_guess_fft / stable_f0)
            print(f"   ↔ Diff vs YIN: {diff_cents:+.1f}¢")
    else:
        print("🔍 Guess FFT: no candidate")

    # --- Étape 5b : Guess pattern-based (indépendant de expected_freq)
    guess_pattern = guess_f0_pattern(signal, sr)
    if guess_pattern.f0:
        print(f"🔍 Guess Pattern: f0={guess_pattern.f0:.2f}Hz "
              f"(conf={guess_pattern.confidence:.2f})")
        for n, freq, err in guess_pattern.matched:
            print(f"   ↳ Harm {n}: {freq:.2f}Hz ({err:+.1f}¢)")
        if stable_f0 > 0:
            diff_cents = 1200 * np.log2(guess_pattern.f0 / stable_f0)
            print(f"   ↔ Diff vs YIN: {diff_cents:+.1f}¢")
    else:
        print("🔍 Guess Pattern: no candidate")
    
    # --- Étape 5c : Guess fusion (combine FFT & Pattern, optionnellement guidé par expected)
    # On passe un hint MIDI si tu l’as sous la main; sinon laisse None.
    # Ici on calcule un hint depuis expected_freq pour aider la fusion.
    midi_hint = int(round(69 + 12 * np.log2(expected_freq / 440.0)))
    guess_fused = guess_f0_fusion(signal, sr, midi_hint=midi_hint)

    if guess_fused.f0:
        print(
            f"🔀 Guess Fusion: f0={guess_fused.f0:.2f}Hz "
            f"(conf={guess_fused.confidence:.2f})"
        )
        if stable_f0 > 0:
            diff_cents = 1200 * np.log2(guess_fused.f0 / stable_f0)
            print(f"   ↔ Diff vs YIN: {diff_cents:+.1f}¢")
    else:
        print("🔀 Guess Fusion: no candidate")

    # --- Retour final (YIN reste la référence pour f0)
    return NoteAnalysisResult(
        note_name=note_name,
        valid=yin_valid,
        f0=stable_f0 if stable_f0 > 0 else None,
        confidence=mode_rate,
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
    )