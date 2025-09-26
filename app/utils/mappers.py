# app/utils/mappers.py

from pytune_dsp.types.dataclasses import NoteAnalysisResult as DSPNoteResult
from pytune_dsp.types.schemas import NoteAnalysisResult as ApiNoteResult, GuessNoteResult


def map_note_result(dsp_result: DSPNoteResult) -> ApiNoteResult:
    return ApiNoteResult(
        note_name=dsp_result.note_name,
        valid=dsp_result.valid,
        f0=dsp_result.f0,
        confidence=dsp_result.confidence,
        deviation_cents=dsp_result.deviation_cents,
        expected_freq=dsp_result.expected_freq,
        harmonics=dsp_result.harmonics,
        partials=dsp_result.partials,
        inharmonicity=[] if dsp_result.inharmonicity is None else dsp_result.inharmonicity,
        spectral_fingerprint=[] if dsp_result.spectral_fingerprint is None else dsp_result.spectral_fingerprint.tolist(),
        harmonic_spectrum_raw=[] if dsp_result.harmonic_spectrum_raw is None else dsp_result.harmonic_spectrum_raw.tolist(),
        harmonic_spectrum_norm=[] if dsp_result.harmonic_spectrum_norm is None else dsp_result.harmonic_spectrum_norm.tolist(),
        inharmonicity_avg=dsp_result.inharmonicity_avg,
        B_estimate=dsp_result.B_estimate,
        guessed_note=(
            None if dsp_result.guessed_note is None else
            GuessNoteResult(
                midi=dsp_result.guessed_note.midi,
                f0=dsp_result.guessed_note.f0,
                confidence=dsp_result.guessed_note.confidence,
                method=dsp_result.guessed_note.method,
            )
        )
    )