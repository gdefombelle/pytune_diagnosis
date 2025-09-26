from typing import List
import numpy as np
from pytune_dsp.types.dataclasses import NoteCaptureMeta, NoteAnalysisResult

# Fake analyzer pour le moment
def analyze_expected_note(meta: NoteCaptureMeta, audio_bytes: bytes) -> NoteAnalysisResult:
    """
    Analyse simulée d'une note de piano.
    Pour l'instant on renvoie des données factices mais cohérentes.
    """

    # Convertir les bytes en numpy (float32)
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as e:
        return NoteAnalysisResult(
            midi=meta.note_expected,
            noteName=f"Note_{meta.note_expected}",
            error=f"Impossible de décoder l'audio: {e}"
        )

    # Longueur utile en secondes
    duration = len(audio_array) / meta.sample_rate

    # Fake f0 → 440Hz si A4, sinon proportionnel au midi
    fake_f0 = 440.0 if meta.note_expected == 69 else 27.5 * (2 ** ((meta.note_expected - 21) / 12))

    # Fake harmonics
    harmonics: List[float] = [fake_f0 * i for i in range(1, 6)]

    return NoteAnalysisResult(
        midi=meta.note_expected,
        noteName=f"Note_{meta.note_expected}",
        f0=round(fake_f0, 2),
        harmonics=harmonics,
        inharmonicity=0.0001 * (meta.note_expected % 10),  # juste pour varier
        detuned_cents=(meta.note_expected % 5) - 2,        # -2 à +2 cents
        error=None
    )