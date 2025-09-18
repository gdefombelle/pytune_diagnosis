# src/models/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List


class NoteCaptureMeta(BaseModel):
    note_expected: int = Field(..., description="MIDI number of expected note")
    sample_rate: int = Field(..., description="Actual sample rate of the stream (Hz)")
    channels: int = Field(1, description="Number of channels (default mono)")
    dtype: str = Field("float32", description="Data type of audio buffer")
    length: int = Field(..., description="Number of samples in buffer")

    @validator("sample_rate")
    def validate_sample_rate(cls, v):
        if v not in (44100, 48000, 88200, 96000):
            raise ValueError(f"Unsupported sample rate: {v}")
        return v


class NoteAnalysisResult(BaseModel):
    midi: int
    noteName: str
    f0: Optional[float] = None
    harmonics: Optional[List[float]] = None
    inharmonicity: Optional[float] = None
    detuned_cents: Optional[float] = None
    error: Optional[str] = None