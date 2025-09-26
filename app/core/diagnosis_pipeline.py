# pytune_dsp/analysis/analyze.py
import numpy as np
from pytune_dsp.types.schemas import NoteAnalysisResult, GuessNoteResult
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
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.utils.note_utils import freq_to_midi, midi_to_freq, freq_to_note


def analyze_note(
    note_name: str,
    expected_freq: float,
    signal: np.ndarray,
    sr: int,
    compute_inharm: bool = True,
    compute_response_data: bool = True,
):
    # --- Étape 0 : Expected (référence)
    midi_expected = freq_to_midi(expected_freq)
    print(f"🎹 Expected: {note_name} (MIDI {midi_expected}, {expected_freq:.2f} Hz)")

    # --- Étape 1 : détection brute de la note via Pattern ---
    guess_pattern = guess_f0_pattern(signal, sr)
    if guess_pattern.f0:
        midi_guess = freq_to_midi(guess_pattern.f0)
        guessed_note_name = freq_to_note(guess_pattern.f0)
        guessed_note = GuessNoteResult(
            midi=midi_guess,
            f0=guess_pattern.f0,
            confidence=guess_pattern.confidence,
            method="pattern",
        )
        print(
            f"🎹 Guessed (pattern): {guessed_note_name} "
            f"(MIDI {midi_guess}, f0≈{guess_pattern.f0:.2f} Hz, "
            f"conf={guess_pattern.confidence:.2f})"
        )
    else:
        guessed_note_name = note_name
        guessed_note = GuessNoteResult(
            midi=None, f0=None, confidence=0.0, method="none"
        )
        print("🎹 Guessed (pattern): none")

    # --- Étape 2 : YIN tracking restreint autour de la freq devinée ---
    yin_center_freq = guess_pattern.f0 if guess_pattern.f0 else expected_freq
    f0s = yin_track(signal, sr, yin_center_freq, semitones=0.5)
    stable_f0, mode_rate = stable_f0_detection(f0s)

    # Étape 3 : Validation YIN
    yin_valid = True
    if stable_f0 < 20 or mode_rate < 0.4:
        yin_valid = False

    deviation_cents = (
        1200 * np.log2(stable_f0 / expected_freq)
        if stable_f0 > 0 and expected_freq
        else None
    )
    if deviation_cents is not None and abs(deviation_cents) > 50:
        yin_valid = False

    # Étape 4 : partiels et inharmonicité
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

    # Étape 5 : empreinte spectrale
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum_norm = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    fingerprint = spectrum_norm[:512]

    raw_fp, norm_fp = harmonic_spectrum_fft(
        signal,
        sr,
        stable_f0 if stable_f0 > 0 else expected_freq,
        nb_harmonics=8,
    )

    # Étape 6 : logs secondaires FFT & Fusion (debug only)
    f0_fft, conf_fft = guess_f0_fft(signal, sr, expected_freq)
    if f0_fft:
        midi_fft = freq_to_midi(f0_fft)
        print(
            f"🔍 Guessed (fft): {freq_to_note(f0_fft)} "
            f"(MIDI {midi_fft}, f0≈{f0_fft:.2f} Hz, conf={conf_fft:.2f})"
        )

    guess_fusion = guess_f0_fusion(signal, sr, None)
    if guess_fusion.f0:
        midi_fus = freq_to_midi(guess_fusion.f0)
        print(
            f"🔍 Fusion result: {freq_to_note(guess_fusion.f0)} "
            f"(MIDI {midi_fus}, f0≈{guess_fusion.f0:.2f} Hz, "
            f"conf={guess_fusion.confidence:.2f})"
        )

    # Étape 7 : response (long sustain)
    response_data = None
    if compute_response_data and len(signal) / sr >= 3.0:
        response_data = compute_response(
            signal, sr, stable_f0 if stable_f0 > 0 else expected_freq
        )

    # --- Retour final ---
    return NoteAnalysisResult(
        note_name=guessed_note_name,
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
        guessed_note=guessed_note,
        response=response_data,
    )