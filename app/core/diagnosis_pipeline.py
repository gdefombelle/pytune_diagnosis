# pytune_diagnosis/app/core/diagnosis_pipeline.py
import numpy as np
from pathlib import Path
import csv
import time
from concurrent.futures import ThreadPoolExecutor

from pytune_dsp.types.schemas import NoteAnalysisResult
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import compute_inharmonicity_avg, estimate_B
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.utils.note_utils import freq_to_midi
from pytune_dsp.utils.serialize import safe_asdict

# Pitch detection engines
from pytune_dsp.analysis.pitch_detection_librosa import guess_note_librosa
from pytune_dsp.analysis.pitch_detection_essentia import guess_note_essentia
from pytune_dsp.analysis.unison import analyze_unison
from pytune_dsp.analysis.f0_HP import compute_f0_HP
from pytune_dsp.analysis.pitch_detection_pfd import estimate_f0_pfd_numba

DEBUG_CSV_PATH = Path("GUESS_NOTES_RESULTS.csv")

# ──────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────
def dump_guesses_to_csv(note_name: str, guesses: dict):
    with DEBUG_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        row = [note_name]
        for k, g in guesses.items():
            f0 = getattr(g, "f0", None)
            conf = getattr(g, "confidence", None)
            if f0:
                row.append(f"{k}:{f0:.2f}Hz({(conf or 0):.2f})")
            else:
                row.append(f"{k}:none")
        writer.writerow(row)


def cents_between(f1: float, f2: float) -> float:
    return 1200.0 * np.log2(max(f1, 1e-12) / max(f2, 1e-12))


def fmt_val(x, digits=2):
    """Formatte proprement tout type de valeur pour les logs."""
    if isinstance(x, (float, int)):
        return f"{x:.{digits}f}"
    if isinstance(x, (tuple, list, np.ndarray)):
        return "[" + ", ".join(fmt_val(v, digits) for v in x) + "]"
    return str(x)


# ──────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE
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
    if sr < 44100:
        raise ValueError(f"Sample rate too low ({sr} Hz). Minimum 44.1 kHz required.")

    midi_expected = freq_to_midi(expected_freq)
    print(f"🎹 Expected {note_name} (MIDI {midi_expected}, {expected_freq:.2f} Hz)")

    def timed(func, *args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        dur_ms = (time.perf_counter() - start) * 1000.0
        return res, dur_ms

    # ───────────── STEP 1 : f₀ multi-engine ─────────────
    start_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_lib = ex.submit(timed, guess_note_librosa, signal, sr, True)
        fut_ess = ex.submit(timed, guess_note_essentia, signal, sr, True)
        fut_pfd = ex.submit(timed, estimate_f0_pfd_numba, signal, sr)
        (guess_lib, time_lib) = fut_lib.result()
        (guess_ess, time_ess) = fut_ess.result()
        (res_pfd, time_pfd) = fut_pfd.result()
    total_time = (time.perf_counter() - start_parallel) * 1000.0

    seq_gain = (time_lib + time_ess + time_pfd) / max(total_time, 1e-9)
    print(f"⏱️ librosa: {time_lib:.1f} ms | essentia: {time_ess:.1f} ms | PFD: {time_pfd:.1f} ms")
    print(f"⚙️ Total parallel time: {total_time:.1f} ms (gain ≈ {seq_gain:.2f}× vs sequential)")

    # Logs PFD
    if isinstance(res_pfd, dict):
        print(f"[PFD] f₀={fmt_val(res_pfd.get('f0'))} Hz | B={fmt_val(res_pfd.get('B'))} | "
              f"quality={fmt_val(res_pfd.get('quality'))} | method=PFD-numba")
    else:
        print(f"[PFD] → Aucun résultat valide ({type(res_pfd)})")

    # ───────────── STEP 2 : sélection / fusion ─────────────
    class PFDGuess:
        def __init__(self, f0, confidence=0.9, method="pfd"):
            self.f0 = f0
            self.method = method
            self.confidence = confidence

    guesses = {"librosa": guess_lib, "essentia": guess_ess}
    if isinstance(res_pfd, dict):
        guesses["pfd"] = PFDGuess(f0=res_pfd.get("f0", 0.0))

    valid_guesses = [g for g in guesses.values() if hasattr(g, "f0") and getattr(g, "f0", 0) > 0]
    if not valid_guesses:
        print("⚠️ Aucun f₀ détecté — signal trop faible ou bruité.")
        return NoteAnalysisResult(note_name=note_name, valid=False, f0=None, confidence=0.0, expected_freq=expected_freq)

    best_guess = max(valid_guesses, key=lambda g: getattr(g, "confidence", 0.0))

    if "pfd" in guesses:
        try:
            if abs(cents_between(guesses["pfd"].f0, best_guess.f0)) < 10:
                best_guess.confidence = min(1.0, getattr(best_guess, "confidence", 0.0) + 0.1)
        except Exception:
            pass

    f0_est = float(getattr(best_guess, "f0", 0.0))
    deviation_cents = 1200 * np.log2(f0_est / expected_freq) if expected_freq > 0 else None
    valid = (f0_est > 20.0) and (getattr(best_guess, "confidence", 0.0) >= 0.30) and (abs(deviation_cents or 0) < 100)

    print(f"🌟 Selected: {getattr(best_guess, 'method', 'unknown')} | "
          f"f0={f0_est:.3f} Hz | conf={getattr(best_guess, 'confidence', 0.0):.2f}")

    # ───────────── STEP 3 : analyse d’unisson ─────────────
    unison = analyze_unison(signal=signal, sr=sr, f0_est=f0_est,
                            midi=int(round(midi_expected)), piano_type=piano_type, era=era)

    posterior = getattr(unison, "posterior", None)
    posterior_str = ", ".join(f"{k}:{fmt_val(v, 3)}" for k, v in posterior.items()) if isinstance(posterior, dict) else str(posterior)

    print(f"🎛 Unison → posterior={posterior_str} | det={fmt_val(getattr(unison, 'detected_n_components', 'n/a'))} | "
          f"beat≈{fmt_val(getattr(unison, 'beat_hz_estimate', 'n/a'))}Hz | "
          f"severity={fmt_val(getattr(unison, 'severity', 'n/a'))} | "
          f"HP={getattr(unison, 'recommend_f0_hp', 'none')} ({fmt_val(getattr(unison, 'confidence', 'n/a'))})")

    # ───────────── STEP 4 : f₀ haute précision / PFD synergy ─────────────
    f0_refined = f0_est
    hp_rec = getattr(unison, "recommend_f0_hp", "none")

    if hp_rec == "global":
        f0_hp, dur_hp = timed(compute_f0_HP, signal, sr, f0_seed=f0_est)
        print(f"🎯 f₀_HP (global): {fmt_val(f0_hp, 4)} Hz ({fmt_val(dur_hp, 1)} ms)")
        f0_hp_val = f0_hp[0] if isinstance(f0_hp, tuple) else f0_hp
        if isinstance(res_pfd, dict) and abs(f0_hp_val - res_pfd.get("f0", 0)) < 0.5:
            f0_refined = res_pfd["f0"]
            print("🧠 f₀_PFD retenu (écart < 0.5 Hz, cohérent avec HP)")
        else:
            f0_refined = f0_hp_val
    else:
        print("⚙️ f₀_HP skipped (unstable unison or low confidence)")

    # ───────────── STEP 5 : Partiels & inharmonicité ─────────────
    if isinstance(f0_refined, (tuple, list, np.ndarray)):
        f0_refined = float(f0_refined[0])

    harmonics, partials, inharmonicity = compute_partials_fft_peaks(
        signal=signal, sr=sr, f0_ref=f0_refined,
        nb_partials=8, search_width_cents=60.0, pad_factor=2)

    partials_hz = [float(p[0]) for p in partials]
    if compute_inharm:
        inharm_avg = compute_inharmonicity_avg(inharmonicity)
        B_pfd_val = res_pfd.get("B") if isinstance(res_pfd, dict) else None
        B_fft = estimate_B(f0_refined, partials_hz) if partials_hz else None
        B_final = B_pfd_val if (isinstance(B_pfd_val, (float, int)) and abs(B_pfd_val) < 1e-2) else B_fft
    else:
        inharm_avg, B_final, B_pfd_val = None, None, None

    print(f"🔍 Inharmonicity → avg={fmt_val(inharm_avg, 3)} | B_PFD={fmt_val(B_pfd_val, 3)} | "
          f"B_final={fmt_val(B_final, 3)} | f₀_PFD={fmt_val(res_pfd.get('f0') if isinstance(res_pfd, dict) else 'n/a')} "
          f"Hz | quality={fmt_val(res_pfd.get('quality') if isinstance(res_pfd, dict) else 'n/a')}")

    # ───────────── STEP 6 : Spectral fingerprint ─────────────
    spectrum = np.abs(np.fft.rfft(signal))
    max_spec = float(np.max(spectrum)) if spectrum.size else 0.0
    spectrum_norm = spectrum / max_spec if max_spec > 0 else spectrum
    fingerprint = spectrum_norm[:512]

    raw_fp, norm_fp = harmonic_spectrum_fft(signal, sr, f0_refined, nb_harmonics=8)

    # ───────────── STEP 7 : Response optionnelle ─────────────
    response_data = None
    if compute_response_data and (len(signal) / sr) >= 3.0:
        response_data = compute_response(signal, sr, f0_refined)

    # ───────────── Nettoyage JSON-safe ─────────────
    def to_list_safe(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [to_list_safe(v) for v in x]
        return x

    harmonics = to_list_safe(harmonics)
    partials_hz = to_list_safe(partials_hz)
    inharmonicity = to_list_safe(inharmonicity)
    fingerprint = to_list_safe(fingerprint)
    raw_fp = to_list_safe(raw_fp)
    norm_fp = to_list_safe(norm_fp)
    response_data = to_list_safe(response_data)

    guesses_return = {k: safe_asdict(v) for k, v in guesses.items() if hasattr(v, "__dict__")}
    if isinstance(res_pfd, dict):
        guesses_return["pfd_raw"] = res_pfd

    return NoteAnalysisResult(
        midi=int(round(midi_expected)),
        note_name=note_name,
        valid=bool(valid),
        f0=float(f0_refined),
        confidence=float(getattr(best_guess, "confidence", 0.0)),
        deviation_cents=float(deviation_cents) if deviation_cents is not None else None,
        expected_freq=float(expected_freq),
        harmonics=harmonics,
        partials=partials_hz,
        inharmonicity=inharmonicity if compute_inharm else None,
        inharmonicity_avg=inharm_avg,
        B_estimate=B_final,
        B_pfd=res_pfd.get("B") if isinstance(res_pfd, dict) else None,
        f0_pfd=res_pfd.get("f0") if isinstance(res_pfd, dict) else None,
        quality_pfd=res_pfd.get("quality") if isinstance(res_pfd, dict) else None,
        spectral_fingerprint=fingerprint,
        harmonic_spectrum_raw=raw_fp,
        harmonic_spectrum_norm=norm_fp,
        guessed_note=safe_asdict(best_guess),
        guesses=guesses_return,
        response=response_data,
        unison=safe_asdict(unison),
    )