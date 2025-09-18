"""
diagnosis_pipeline.py
=====================
Pipeline principal pour l’analyse d’une note isolée (diagnostic PyTune).
"""

import numpy as np
import librosa

from pytune_dsp.analysis.f0_analysis import stable_f0_detection, calculate_f0_measurements
from pytune_dsp.analysis.f0_harmonics import estimate_f0_and_harmonics_using_fft
from pytune_dsp.analysis.partials import compute_partials_recursive_yin, calculate_inharmonicity
from pytune_dsp.types.analysis import NoteMeasurements
from pytune_dsp.core import generate_equal_tempered

# Résultat structuré
from dataclasses import dataclass, field
from typing import List


@dataclass
class NoteAnalysisResult:
    note_name: str
    valid: bool
    f0: float = 0.0
    confidence: float = 0.0
    harmonics: List[float] = field(default_factory=list)
    inharmonicity: List[float] = field(default_factory=list)
    spectral_fingerprint: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------- Vérification du signal ----------

def check_single_note(signal: np.ndarray, sr: int, expected_note: str) -> bool:
    """
    Vérifie que le signal correspond bien à une note isolée.
    - Utilise librosa.yin avec une fenêtre centrée sur la note attendue
    - Vérifie qu’il n’y a pas plusieurs fondamentales concurrentes
    """
    # Construire un clavier de référence
    scale = generate_equal_tempered()

    # Fréquence attendue en tempérament égal
    target_freq = scale[expected_note]

    # Demi-ton haut/bas (≈100 cents)
    fmin = target_freq / (2 ** (1 / 24))  # -0.5 ton
    fmax = target_freq * (2 ** (1 / 24))  # +0.5 ton

    # Détection YIN
    f0s = librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sr)

    # F0 stable et proportion
    stable_f0, mode_rate = stable_f0_detection(f0s)

    # Heuristique : fréquence trouvée proche de la cible, confiance suffisante
    if stable_f0 < 20 or mode_rate < 0.5:
        return False
    deviation_cents = 1200 * np.log2(stable_f0 / target_freq)
    if abs(deviation_cents) > 50:  # plus d’un demi-ton d’écart
        return False

    return True


# ---------- Pipeline principal ----------

def analyze_note(note_name: str, signal: np.ndarray, sr: int) -> NoteAnalysisResult:
    """
    Analyse une note isolée : vérification, f0, harmoniques, inharmonicité, empreinte spectrale.
    """
    # Étape 1 : validation
    is_valid = check_single_note(signal, sr, note_name)
    if not is_valid:
        return NoteAnalysisResult(note_name=note_name, valid=False)

    # Étape 2 : f0 précis
    f0s = librosa.yin(signal, fmin=20, fmax=5000, sr=sr)
    stable_f0, mode_rate = stable_f0_detection(f0s)

    # Étape 3 : partiels et inharmonicité
    harmonics, partials, inharm = compute_partials_recursive_yin(
        None,  # TODO: Keyboard (tempéré/étiré)
        nb_of_partials=10,
        note_deviation_analysis=None,  # TODO: wrapper NoteMeasurements
        sr=sr,
    )

    # Étape 4 : empreinte spectrale
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum_norm = spectrum / np.max(spectrum)
    fingerprint = spectrum_norm[:512]  # ex: garder les 512 premiers bins

    return NoteAnalysisResult(
        note_name=note_name,
        valid=True,
        f0=stable_f0,
        confidence=mode_rate,
        harmonics=list(partials),
        inharmonicity=list(inharm),
        spectral_fingerprint=fingerprint,
    )