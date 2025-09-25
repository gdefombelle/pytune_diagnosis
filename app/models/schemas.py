# src/models/schemas.py
from pydantic import BaseModel, Field, field_validator

from typing import Optional, List, Tuple


class NoteCaptureMeta(BaseModel):
    note_expected: int = Field(..., description="MIDI number of expected note")
    sample_rate: int = Field(..., description="Actual sample rate of the stream (Hz)")
    channels: int = Field(1, description="Number of channels (default mono)")
    dtype: str = Field("float32", description="Data type of audio buffer")
    length: int = Field(..., description="Number of samples in buffer")
    compute_inharm: bool = Field(
        True, description="Whether to compute inharmonicity (False for high notes > C6)"
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in (44100, 48000, 88200, 96000):
            raise ValueError(f"Unsupported sample rate: {v}")
        return v


class NoteAnalysisResult(BaseModel):
    note_name: str
    valid: bool
    f0: Optional[float] = None
    confidence: Optional[float] = None
    deviation_cents: Optional[float] = None   # écart vs expected
    expected_freq: Optional[float] = None     # traçabilité

    harmonics: List[float] = []               # fréquences théoriques k*f0
    partials: List[float] = []                # fréquences mesurées (Hz)
    inharmonicity: List[float] = []           # déviation (cents, un par partiel)

    spectral_fingerprint: List[float] = []    # compact, normalisé (hash-like)

    # Ajouts spectre harmonique
    harmonic_spectrum_raw: List[Tuple[float, float]] = []   # (freq, amplitude brute)
    harmonic_spectrum_norm: List[Tuple[float, float]] = []  # (freq, amplitude normalisée 0–1)

    # 🔹 Nouveaux ajouts utiles pour le frontend
    inharmonicity_avg: Optional[float] = None  # moyenne pondérée/simplifiée en cents
    B_estimate: Optional[float] = None         # coefficient B global estimé