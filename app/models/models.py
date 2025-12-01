# fields we never want to send to the frontend live
_SENSITIVE_FIELDS = {
    "spectral_fingerprint",
    "harmonic_spectrum_raw",
    "harmonic_spectrum_norm",
    "response",
    "guesses",
    "streams_debug",
    "time_librosa_ms",
    "time_essentia_ms",
    "time_pfd_ms",
    "time_parallel_ms",
}

def slim_note_analysis(result):
    """Return a massively slimmed-down dict version of NoteAnalysisResult 
    for SSE/live streaming."""
    if result is None:
        return None

    # Convert to a dict
    data = result.dict() if hasattr(result, "dict") else dict(result)

    # Remove heavy / debug fields
    for field in _SENSITIVE_FIELDS:
        if field in data:
            data[field] = None

    return data