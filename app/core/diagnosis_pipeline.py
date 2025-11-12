import math
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
    use_essentia: bool = False,   # ✅ nouveau paramètre (facultatif)
    use_librosa: bool = False,    # ✅ nouveau paramètre (facultatif)
    # ── contexte (neutre par défaut) ───────────────────────────
    context_expected_weight: float = 0.0,     # 0 = off
    context_recent_weight: float = 0.0,       # 0 = off
    recent_notes_hz: list[float] | None = None,  # dernières f0 jouées (Hz)
    context_cents_sigma: float = 50.0,        # portée de l’effet (≈ 50 cents)
) -> NoteAnalysisResult:

    midi_expected = freq_to_midi(expected_freq)
    print(f"\n🎹  {bold(note_name)} expected → MIDI {midi_expected} | {expected_freq:.2f} Hz")

    # ───── Parallel pitch estimations ─────
    def timed(func, *args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        return res, (time.perf_counter() - start) * 1000.0

    start_all = time.perf_counter()

    # ✅ nombre de workers selon les switches
    n_workers = 2 + int(use_librosa) + int(use_essentia)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        fut_lib, fut_ess = None, None

        # --- Librosa (facultatif) ---
        if use_librosa:
            fut_lib = ex.submit(timed, guess_note_librosa, signal, sr, True)

        # --- HPS & HPS-multi toujours actifs ---
        fut_hps = ex.submit(timed, estimate_f0_hps_wrapper, signal, sr, "auto")
        fut_hpsm = ex.submit(timed, estimate_f0_hps_multi_wrapper, signal, sr)

        # --- Essentia (facultatif) ---
        if use_essentia:
            fut_ess = ex.submit(timed, guess_note_essentia, signal, sr, True)

        # --- Résultats ---
        guess_lib, t_lib = (None, 0.0)
        guess_ess, t_ess = (None, 0.0)
        if use_librosa and fut_lib:
            (guess_lib, t_lib) = fut_lib.result()

        (res_hps, t_hps) = fut_hps.result()
        (res_hpsm, t_hpsm) = fut_hpsm.result()

        if use_essentia and fut_ess:
            (guess_ess, t_ess) = fut_ess.result()

        t_total = (time.perf_counter() - start_all) * 1000.0

    # --- Normaliser la sortie HPS-multi (accepte "best" OU "candidates[0]")
    hpsm_best = None
    if isinstance(res_hpsm, dict):
        if res_hpsm.get("best"):
            hpsm_best = res_hpsm["best"]
        else:
            cands = res_hpsm.get("candidates")
            if isinstance(cands, list) and cands:
                hpsm_best = cands[0]

    # Affichage timings
    print(cyan(
        "⏱ timings → "
        + (f"librosa={t_lib:.1f} ms | " if use_librosa else "")
        + (f"essentia={t_ess:.1f} ms | " if use_essentia else "")
        + f"HPS={t_hps:.1f} ms | multi={t_hpsm:.1f} ms | total={t_total:.1f} ms"
    ))

    # ───── Normalize guesses ─────
    class Guess:
        def __init__(self, f0, conf, method):
            self.f0 = float(f0)
            self.confidence = float(conf)
            self.method = str(method)
    def _cents_delta(f_hz: float, ref_hz: float) -> float:
        if f_hz <= 0 or ref_hz <= 0:
            return 1e9
        return 1200.0 * math.log2(f_hz / ref_hz)

    def _gauss(x: float, sigma: float) -> float:
        # pic = 1 à x=0, décroît en gaussienne ; borné ∈ (0,1]
        if sigma <= 0:
            return 1.0
        return math.exp(-0.5 * (x / sigma) ** 2)

    proposals = {}

    # --- Librosa (facultatif) ---
    if use_librosa and hasattr(guess_lib, "f0") and guess_lib.f0 > 0:
        proposals["librosa"] = Guess(
            guess_lib.f0, getattr(guess_lib, "confidence", 0.0), "librosa"
        )

    # --- Essentia (facultatif) ---
    if use_essentia and hasattr(guess_ess, "f0") and guess_ess.f0 > 0:
        proposals["essentia"] = Guess(
            guess_ess.f0, getattr(guess_ess, "confidence", 0.0), "essentia"
        )

    # --- HPS (wrapper "single") ---
    if isinstance(res_hps, dict) and res_hps.get("f0", 0) > 0:
        proposals["hps"] = Guess(
            res_hps["f0"], res_hps.get("quality", 0.0), "hps"
        )

    # --- HPS-multi (best normalisé) ---
    if hpsm_best and hpsm_best.get("f0", 0) > 0:
        proposals["hps_multi"] = Guess(
            hpsm_best["f0"],
            hpsm_best.get("conf", hpsm_best.get("quality", 0.0)),
            "hps_multi"
        )
        print(gray(
            f"[HPS-multi used] best → {hpsm_best['f0']:.2f} Hz "
            f"(q={hpsm_best.get('conf', hpsm_best.get('quality', 0.0)):.2f})"
        ))

    # Vérification
    if not proposals:
        print(red("⚠ Aucun moteur n’a produit de f₀ valide."))
        return NoteAnalysisResult(
            note_name=note_name,
            valid=False,
            f0=None,
            confidence=0.0,
            expected_freq=expected_freq
        )

    recent_ref_hz = None
    if context_recent_weight > 0.0 and recent_notes_hz:
        # médiane robuste des dernières notes jouées
        try:
            arr = [float(f) for f in recent_notes_hz if f and f > 0]
            if arr:
                arr.sort()
                mid = len(arr) // 2
                recent_ref_hz = arr[mid] if len(arr) % 2 == 1 else 0.5 * (arr[mid-1] + arr[mid])
        except Exception:
            recent_ref_hz = None

    # ───── Anti-octave piloté par HPS ─────
    if "hps" in proposals:
        hps_f0 = proposals["hps"].f0
        for g in proposals.values():
            r = abs(np.log2(g.f0 / hps_f0))
            if 0.45 < r < 0.55:  # ~1 octave
                g.confidence *= 0.60  # pénalité plus forte
            if r < 0.03:  # proche du HPS (±36 cents)
                g.confidence *= 1.10

    # ───── Contexte optionnel (neutre si poids=0) ─────
    if context_expected_weight > 0.0 and expected_freq > 0:
        for g in proposals.values():
            dc = abs(_cents_delta(g.f0, expected_freq))
            # facteur ∈ (1, 1+poids], max au voisinage de la note attendue
            g.confidence *= (1.0 + context_expected_weight * _gauss(dc, context_cents_sigma))

    if context_recent_weight > 0.0 and recent_ref_hz and recent_ref_hz > 0:
        for g in proposals.values():
            dc = abs(_cents_delta(g.f0, recent_ref_hz))
            g.confidence *= (1.0 + context_recent_weight * _gauss(dc, context_cents_sigma))

    # ───── Sélection finale ─────
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

    # HPS-multi subresults (si dispo)
    sub_hpsm = None
    if hpsm_best:
        sub_hpsm = {
            "hps_multi": {
                "f0": float(hpsm_best.get("f0", 0.0)),
                "conf": float(hpsm_best.get("conf", hpsm_best.get("quality", 0.0))),
                "score": float(hpsm_best.get("score", 0.0)),
            }
        }

    sub_all = {}
    for s in (sub_lib, sub_ess, sub_hps, sub_hpsm):
        if s:
            sub_all.update(s)

    guessed_note = {
        "f0": f0_est,
        "confidence": best.confidence,
        "method": best.method,
        "envelope_band": getattr(guess_ess, "envelope_band", "unknown"),
        "subresults": sub_all,
        "midi": int(round(freq_to_midi(f0_est))),
        "note_name": freq_to_note(f0_est),
    }

    guesses = {
        "librosa": {"f0": getattr(guess_lib, "f0", 0.0), "confidence": getattr(guess_lib, "confidence", 0.0),
                    "method": "librosa", "subresults": sub_lib},
        "essentia": {"f0": getattr(guess_ess, "f0", 0.0), "confidence": getattr(guess_ess, "confidence", 0.0),
                     "method": "essentia", "subresults": sub_ess},
        "hps": {"f0": res_hps.get("f0", 0.0) if isinstance(res_hps, dict) else None,
                "confidence": res_hps.get("quality", 0.0) if isinstance(res_hps, dict) else None,
                "method": "HPS", "subresults": sub_hps},
        "hps_multi": {"f0": hpsm_best.get("f0", 0.0) if hpsm_best else 0.0,
                      "confidence": hpsm_best.get("conf", hpsm_best.get("quality", 0.0)) if hpsm_best else 0.0,
                      "method": "HPS-multi", "subresults": sub_hpsm},
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

    # ───── Append to CSV log ─────
    try:
        from datetime import datetime
        from app.utils.csv_logger import append_diagnosis_row

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def fmt(x, digits=4):
            try:
                return f"{float(x):.{digits}f}"
            except Exception:
                return ""

        def fmt_exp(x):
            try:
                return f"{float(x):.3e}"
            except Exception:
                return ""

        # Sélecteur robuste pour HPS-multi (best/candidates)
        def pick_hpsm_best(d):
            if not isinstance(d, dict):
                return {}
            if d.get("best"):
                return d["best"]
            c = d.get("candidates")
            return c[0] if isinstance(c, list) and c else {}

        hpsm_csv = pick_hpsm_best(res_hpsm)

        append_diagnosis_row({
            "timestamp": timestamp,
            "expected_note": freq_to_note(expected_freq),
            "expected_freq": fmt(expected_freq),
            "f0_seed_note": freq_to_note(f0_est),

            "librosa_f0": fmt(getattr(guess_lib, "f0", 0.0)),
            "librosa_conf": fmt(getattr(guess_lib, "confidence", 0.0)),

            "essentia_f0": fmt(getattr(guess_ess, "f0", 0.0)),
            "essentia_conf": fmt(getattr(guess_ess, "confidence", 0.0)),

            "hps_f0": fmt(res_hps.get("f0", 0.0) if isinstance(res_hps, dict) else 0.0),
            "hps_q": fmt(res_hps.get("quality", 0.0) if isinstance(res_hps, dict) else 0.0),

            "hpsm_f0": fmt(hpsm_csv.get("f0", 0.0)),
            "hpsm_q":  fmt(hpsm_csv.get("conf", hpsm_csv.get("quality", 0.0))),

            "final_f0": fmt(f0_ref),
            "final_conf": fmt(best.confidence),
            "delta_cents": fmt(dev_cents),
            "B": fmt_exp(B_final) if B_final is not None else "",
            "n_cordes": getattr(unison, "detected_n_components", None),
            "beat_hz": fmt(getattr(unison, "beat_hz_estimate", 0)),
            "valid": bool(valid),
            "t_total_ms": fmt(t_total, 2),
        })
        print(gray("🧾 logged to diagnosis_results.csv"))
    except Exception as e:
        print(red(f"⚠️  logging error: {e}"))

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