import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from pytune_dsp.types.schemas import NoteAnalysisResult
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import compute_inharmonicity_avg, estimate_B
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.utils.note_utils import freq_to_midi, freq_to_note
from pytune_dsp.utils.serialize import safe_asdict
from pytune_dsp.analysis.pitch_detection_librosa import guess_note_librosa
from pytune_dsp.analysis.pitch_detection_essentia import guess_note_essentia
from pytune_dsp.analysis.unison import analyze_unison
from pytune_dsp.analysis.f0_HP import compute_f0_HP
from pytune_dsp.analysis.hps_seq import (
    estimate_f0_hps_wrapper,
    estimate_f0_hps_multi_wrapper
)


# ──────────────────────────────────────────────
# COLOR HELPERS
# ──────────────────────────────────────────────
def c(text, code):
    return f"\033[{code}m{text}\033[0m"


def cyan(t): return c(t, "96")
def green(t): return c(t, "92")
def yellow(t): return c(t, "93")
def red(t): return c(t, "91")
def gray(t): return c(t, "90")
def bold(t): return c(t, "1")


def fmt_val(x, digits=3):
    if isinstance(x, (float, int)):
        return f"{x:.{digits}f}"
    return str(x)


# ──────────────────────────────────────────────
# MAIN ANALYSIS
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

    midi_expected = freq_to_midi(expected_freq)
    print(f"\n🎹  {bold(note_name)} expected → MIDI {midi_expected} | {expected_freq:.2f} Hz")

    # ───── Parallel pitch estimations ─────
    def timed(func, *args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        return res, (time.perf_counter() - start) * 1000.0

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_lib = ex.submit(timed, guess_note_librosa, signal, sr, True)
        fut_ess = ex.submit(timed, guess_note_essentia, signal, sr, True)
        fut_hps = ex.submit(timed, estimate_f0_hps_wrapper, signal, sr, "auto")
        fut_hpsm = ex.submit(timed, estimate_f0_hps_multi_wrapper, signal, sr)

        (guess_lib, t_lib) = fut_lib.result()
        (guess_ess, t_ess) = fut_ess.result()
        (res_hps,  t_hps) = fut_hps.result()
        (res_hpsm, t_hpsm) = fut_hpsm.result()
    
    # juste après les fut_*.result()
    if isinstance(res_hps, dict) and res_hps.get("f0", 0) > 0:
        print(gray(f"[HPS-wrapper] → {res_hps['f0']:.2f} Hz (q={res_hps.get('quality',0):.2f})"
                f" B={res_hps.get('B',0):.3f}"))

    if isinstance(res_hpsm, dict) and res_hpsm.get("best"):
        hb = res_hpsm["best"]
        print(gray(f"[HPS-multi] best → {hb['f0']:.2f} Hz (q={hb.get('quality',0):.2f})"))

    t_total = (time.perf_counter() - start_all) * 1000.0
    print(cyan(f"⏱  timings → librosa={t_lib:.1f} ms | essentia={t_ess:.1f} ms | "
               f"HPS={t_hps:.1f} ms | multi={t_hpsm:.1f} ms  → total={t_total:.1f} ms"))

    # ───── Normalize guesses ─────
    class Guess:
        def __init__(self, f0, conf, method):
            self.f0 = float(f0)
            self.confidence = float(conf)
            self.method = str(method)

    proposals = {}

    # --- Librosa ---
    if hasattr(guess_lib, "f0") and guess_lib.f0 > 0:
        proposals["librosa"] = Guess(
            guess_lib.f0, getattr(guess_lib, "confidence", 0.0), "librosa"
        )

    # --- Essentia ---
    if hasattr(guess_ess, "f0") and guess_ess.f0 > 0:
        proposals["essentia"] = Guess(
            guess_ess.f0, getattr(guess_ess, "confidence", 0.0), "essentia"
        )

    # --- HPS (wrapper rapide) ---
    if isinstance(res_hps, dict) and res_hps.get("f0", 0) > 0:
        proposals["hps"] = Guess(
            res_hps["f0"], res_hps.get("quality", 0.0), "hps"
        )

    # --- HPS-multi (validation inter-octave / précision) ---
    if isinstance(res_hpsm, dict) and res_hpsm.get("best"):
        hb = res_hpsm["best"]
        if hb.get("f0", 0) > 0:
            proposals["hps_multi"] = Guess(
                hb["f0"], hb.get("quality", 0.0), "hps_multi"
            )
            print(gray(
                f"[HPS-multi used] best → {hb['f0']:.2f} Hz (q={hb.get('quality', 0):.2f})"
            ))

    # --- Vérification globale ---
    if not proposals:
        print(red("⚠ Aucun moteur n’a produit de f₀ valide."))
        return NoteAnalysisResult(
            note_name=note_name,
            valid=False,
            f0=None,
            confidence=0.0,
            expected_freq=expected_freq
        )
    # -- nudge anti-octave piloté par HPS (si dispo)
    if "hps" in proposals:
        hps_f0 = proposals["hps"].f0
        for g in proposals.values():
            r = abs(np.log2(g.f0 / hps_f0))
            # si le candidat est ~à ±1 octave du HPS (r≈1.0) → petite pénalité
            if 0.45 < r < 0.55:
                g.confidence *= 0.85
            # si le candidat est proche du HPS (±36 cents) → petit bonus
            if r < 0.03:
                g.confidence *= 1.05

    best = max(proposals.values(), key=lambda g: g.confidence)
    f0_est = best.f0
    dev_cents = 1200 * np.log2(f0_est / expected_freq)
    valid = (f0_est > 20.0) and (abs(dev_cents) < 100) and (best.confidence >= 0.25)
    print(green(f"🌟 selected: {best.method} → {f0_est:.2f} Hz ({best.confidence:.2f})  Δ={dev_cents:.1f} cents"))

    # ───── Unison analysis ─────
    unison = analyze_unison(signal, sr, f0_est, int(round(midi_expected)), piano_type, era)
    print(gray(f"🎛 unison → n={getattr(unison,'detected_n_components','?')}  "
               f"beat≈{fmt_val(getattr(unison,'beat_hz_estimate','?'))} Hz  "
               f"severity={fmt_val(getattr(unison,'severity','?'))}"))

    f0_ref = f0_est
    if getattr(unison, "recommend_f0_hp", "") == "global":
        f0_hp, dur_hp = timed(compute_f0_HP, signal, sr, f0_seed=f0_est)
        f0_ref = f0_hp[0] if isinstance(f0_hp, tuple) else f0_hp
        print(yellow(f"🎯 refined f₀_HP = {f0_ref:.3f} Hz ({dur_hp:.1f} ms)"))
    else:
        print(gray("⚙ skipped high-precision refinement (low unison confidence)"))

    # ───── Harmonic analysis ─────
    harmonics, partials, inharm = compute_partials_fft_peaks(
        signal, sr, f0_ref, nb_partials=8, search_width_cents=60, pad_factor=2
    )
    part_freqs = [float(p[0]) for p in partials]
    B_final = estimate_B(f0_ref, part_freqs) if compute_inharm else None
    avg_inh = compute_inharmonicity_avg(inharm) if compute_inharm else None
    print(gray(f"🔍 inharmonicity → avg={fmt_val(avg_inh)}  B={fmt_val(B_final)}"))

    # ───── Spectrum fingerprints ─────
    spec = np.abs(np.fft.rfft(signal))
    spec /= (np.max(spec) or 1)
    fingerprint = spec[:512]
    raw_fp, norm_fp = harmonic_spectrum_fft(signal, sr, f0_ref, nb_harmonics=8)

    # ───── Debug subresults for frontend ─────
    def extract_subresults(engine):
        out = {}
        if hasattr(engine, "subresults") and isinstance(engine.subresults, dict):
            for k, v in engine.subresults.items():
                if isinstance(v, dict):
                    out[k.lower()] = {
                        "f0": float(v.get("f0", 0.0)),
                        "conf": float(v.get("conf", v.get("confidence", 0.0))),
                        "score": float(v.get("score", 0.0)),
                    }
        elif isinstance(engine, dict):
            out["hps"] = {
                "f0": float(engine.get("f0", 0.0)),
                "conf": float(engine.get("quality", 0.0)),
                "score": float(engine.get("score", 0.0)),
            }
        return out or None

    sub_lib = extract_subresults(guess_lib)
    sub_ess = extract_subresults(guess_ess)
    sub_hps = extract_subresults(res_hps)
    sub_all = {}
    for s in (sub_lib, sub_ess, sub_hps):
        if s:
            sub_all.update(s)

    guessed_note = {
        "f0": f0_est,
        "confidence": best.confidence,
        "method": best.method,
        "envelope_band": getattr(guess_ess, "envelope_band", "unknown"),
        "subresults": sub_all,
        "midi": int(round(freq_to_midi(f0_est))),
        "note_name": freq_to_note(f0_est),  # à créer juste après
    }

    guesses = {
        "librosa": {"f0": getattr(guess_lib, "f0", 0.0), "confidence": getattr(guess_lib, "confidence", 0.0),
                    "method": "librosa", "subresults": sub_lib},
        "essentia": {"f0": getattr(guess_ess, "f0", 0.0), "confidence": getattr(guess_ess, "confidence", 0.0),
                     "method": "essentia", "subresults": sub_ess},
        "hps": {"f0": res_hps.get("f0", 0.0) if isinstance(res_hps, dict) else None,
                "confidence": res_hps.get("quality", 0.0) if isinstance(res_hps, dict) else None,
                "method": "HPS", "subresults": sub_hps},
    }

    hps_data = None
    if isinstance(res_hps, dict):
        hps_data = {
            "f0": float(res_hps.get("f0", 0.0)),
            "quality": float(res_hps.get("quality", 0.0)),
            "B": float(res_hps.get("B", 0.0)),
            "method": "HPS-wrapper",
        }

    # ───── Response envelope ─────
    response_data = None
    if compute_response_data and (len(signal) / sr) >= 3.0:
        response_data = compute_response(signal, sr, f0_ref)
        print(gray("📈 response curve computed"))

    # ───── Final summary ─────
    print(green(f"✅ DONE — f₀={f0_ref:.2f} Hz | conf={best.confidence:.2f} | "
                f"B={fmt_val(B_final)} | valid={valid}"))

    return NoteAnalysisResult(
        midi=int(round(midi_expected)),
        note_name=note_name,
        valid=bool(valid),
        f0=float(f0_ref),
        confidence=float(best.confidence),
        deviation_cents=float(dev_cents),
        expected_freq=float(expected_freq),

        harmonics=harmonics,
        partials=part_freqs,
        inharmonicity=inharm,
        inharmonicity_avg=avg_inh,
        B_estimate=B_final,

        spectral_fingerprint=fingerprint.tolist(),
        harmonic_spectrum_raw=raw_fp,
        harmonic_spectrum_norm=norm_fp,

        guessed_note=guessed_note,
        guesses=guesses,
        hps=hps_data,
        unison=safe_asdict(unison),
        response=response_data,

        time_librosa_ms=t_lib,
        time_essentia_ms=t_ess,
        time_pfd_ms=t_hps,
        time_parallel_ms=t_total,
    )