import math
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from typing import Any
from pytune_dsp.types.schemas import NoteAnalysisResult, GuessNoteResult
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import compute_inharmonicity_avg, estimate_B
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.utils.note_utils import freq_to_midi, freq_to_note
from pytune_dsp.utils.serialize import safe_asdict
from pytune_dsp.analysis.f0_HP_v2 import compute_f0_HP_v2

# primary seed detectors
from pytune_dsp.analysis.pitch_detection_expected import detect_f0_informed_expected
from pytune_dsp.analysis.yin_partitioned import detect_f0_seed_partitioned
from pytune_dsp.analysis.yin_backend_essentia_frames import yin_backend_essentia_frames

# precision / acoustic analysis
from pytune_dsp.analysis.unison import analyze_unison
from pytune_dsp.analysis.hps_seq import (
    estimate_f0_hps_wrapper,
    estimate_f0_hps_multi_wrapper,
)


# ──────────────────────────────────────────────
# COLOR HELPERS
# ──────────────────────────────────────────────

def c(t, code):
    return f"\033[{code}m{t}\033[0m"


def cyan(t):
    return c(t, "96")


def green(t):
    return c(t, "92")


def yellow(t):
    return c(t, "93")


def red(t):
    return c(t, "91")


def gray(t):
    return c(t, "90")


def bold(t):
    return c(t, "1")


def fmt_val(x, digits=3):
    if isinstance(x, (float, int)):
        return f"{x:.{digits}f}"
    return str(x)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────


def _cents_delta(f: float | None, ref: float | None) -> float:
    if not f or not ref or f <= 0 or ref <= 0:
        return 1e9
    return 1200.0 * math.log2(f / ref)


class SeedGuess:
    def __init__(self, f0: float, confidence: float, method: str, source: str):
        self.f0 = float(f0)
        self.confidence = float(confidence)
        self.method = method
        self.source = source


class PrimarySeedDecision:
    def __init__(
        self,
        selected: SeedGuess | None,
        informed_guess=None,
        informed_confidence: float = 0.0,
        informed_delta_cents: float | None = None,
        informed_accepted: bool = False,
        informed_reject_reason: str | None = None,
        partitioned_f0: float | None = None,
        partitioned_score: float = 0.0,
        partitioned_history: list | None = None,
        fallback_used: bool = False,
    ):
        self.selected = selected
        self.informed_guess = informed_guess
        self.informed_confidence = float(informed_confidence)
        self.informed_delta_cents = informed_delta_cents
        self.informed_accepted = bool(informed_accepted)
        self.informed_reject_reason = informed_reject_reason
        self.partitioned_f0 = partitioned_f0
        self.partitioned_score = float(partitioned_score)
        self.partitioned_history = partitioned_history or []
        self.fallback_used = bool(fallback_used)


class PrecisionRefineDecision:
    def __init__(
        self,
        f0_seed: float,
        f0_refined: float,
        hps_result,
        hps_multi_result,
        hps_best,
        hp_confidence: float | None,
        method: str,
    ):
        self.f0_seed = float(f0_seed)
        self.f0_refined = float(f0_refined)
        self.hps_result = hps_result
        self.hps_multi_result = hps_multi_result
        self.hps_best = hps_best
        self.hp_confidence = hp_confidence
        self.method = method



def _pick_partitioned_score(yin_history: list) -> float:
    if not yin_history:
        return 0.0
    valid_bands = [br for br in yin_history if getattr(br, "f0", None) is not None]
    if not valid_bands:
        return 0.0
    best_br = max(valid_bands, key=lambda br: br.score)
    return float(best_br.score)



def _pick_hps_multi_best(res_hpsm):
    if not isinstance(res_hpsm, dict):
        return None
    if "best" in res_hpsm:
        return res_hpsm["best"]
    if isinstance(res_hpsm.get("candidates"), list) and res_hpsm["candidates"]:
        return res_hpsm["candidates"][0]
    return None



def _extract_subresults(engine):
    if not engine:
        return None

    out = {}
    if hasattr(engine, "subresults") and isinstance(engine.subresults, dict):
        for k, v in engine.subresults.items():
            if isinstance(v, dict):
                out[k.lower()] = {
                    "f0": float(v.get("f0", 0)),
                    "conf": float(v.get("conf", v.get("confidence", 0))),
                    "score": float(v.get("score", 0)),
                }
    elif isinstance(engine, dict):
        out["hps"] = {
            "f0": float(engine.get("f0", 0)),
            "conf": float(engine.get("quality", 0)),
            "score": float(engine.get("score", 0)),
        }

    return out or None



def _detect_primary_seed(
    signal: np.ndarray,
    sr: int,
    expected_freq: float,
    use_informed_expected: bool,
    use_uninformed_partitioned: bool,
    yin_partition_mode: str,
    yin_target_width_oct: float,
    yin_max_depth: int,
    informed_window_cents: float = 300.0,
    informed_accept_confidence: float = 0.45,
    informed_accept_delta_cents: float = 60.0,
):
    def timed(func, *args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        return out, (time.perf_counter() - t0) * 1000

    fut_informed = None
    fut_partitioned = None
    n_workers = max(1, int(use_informed_expected) + int(use_uninformed_partitioned))

    t_block_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        if use_informed_expected:
            fut_informed = ex.submit(
                timed,
                detect_f0_informed_expected,
                signal,
                sr,
                expected_freq,
                informed_window_cents,
                True,
            )

        if use_uninformed_partitioned:
            fut_partitioned = ex.submit(
                timed,
                detect_f0_seed_partitioned,
                signal,
                sr,
                yin_partition_mode,
                yin_backend_essentia_frames,
                yin_target_width_oct,
                yin_max_depth,
            )

        informed_guess, t_informed = (None, 0.0)
        if fut_informed is not None:
            informed_guess, t_informed = fut_informed.result()

        partitioned_f0 = None
        partitioned_history = []
        partitioned_score = 0.0
        t_partitioned = 0.0
        if fut_partitioned is not None:
            partitioned_out, t_partitioned = fut_partitioned.result()
            if partitioned_out is not None:
                partitioned_f0, partitioned_history = partitioned_out
                partitioned_score = _pick_partitioned_score(partitioned_history)

    t_seed_block = (time.perf_counter() - t_block_start) * 1000

    informed_conf = 0.0
    informed_delta = None
    informed_accepted = False
    informed_reject_reason = None
    selected = None
    fallback_used = False

    if use_informed_expected and informed_guess and hasattr(informed_guess, "f0") and informed_guess.f0 > 0:
        informed_conf = max(0.0, float(getattr(informed_guess, "confidence", 0.0)))
        informed_delta = _cents_delta(float(informed_guess.f0), expected_freq)

        if informed_conf < informed_accept_confidence:
            informed_reject_reason = "low_confidence"
        elif abs(informed_delta) > informed_accept_delta_cents:
            informed_reject_reason = "out_of_expected_window"
        else:
            informed_accepted = True
            selected = SeedGuess(
                f0=float(informed_guess.f0),
                confidence=float(informed_conf),
                method="informed_expected",
                source="expected",
            )

    if selected is None and use_uninformed_partitioned and partitioned_f0 and partitioned_f0 > 0:
        fallback_used = bool(use_informed_expected)
        selected = SeedGuess(
            f0=float(partitioned_f0),
            confidence=max(0.35, min(0.99, partitioned_score if partitioned_score > 0 else 0.65)),
            method="yin_partitioned",
            source="uninformed",
        )

    if selected is None and use_informed_expected and informed_guess and hasattr(informed_guess, "f0") and informed_guess.f0 > 0:
        selected = SeedGuess(
            f0=float(informed_guess.f0),
            confidence=float(max(0.0, informed_conf)),
            method="informed_expected_fallback",
            source="expected",
        )
        if informed_reject_reason is None:
            informed_reject_reason = "fallback_no_partitioned_candidate"

    decision = PrimarySeedDecision(
        selected=selected,
        informed_guess=informed_guess,
        informed_confidence=informed_conf,
        informed_delta_cents=informed_delta,
        informed_accepted=informed_accepted,
        informed_reject_reason=informed_reject_reason,
        partitioned_f0=partitioned_f0,
        partitioned_score=partitioned_score,
        partitioned_history=partitioned_history,
        fallback_used=fallback_used,
    )

    timings = {
        "informed_ms": float(t_informed),
        "partitioned_ms": float(t_partitioned),
        "seed_block_ms": float(t_seed_block),
    }
    return decision, timings



def _refine_precision_from_seed(
    signal: np.ndarray,
    sr: int,
    f0_seed: float,
):
    def timed(func, *args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        return out, (time.perf_counter() - t0) * 1000

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_hps = ex.submit(timed, estimate_f0_hps_wrapper, signal, sr, "auto")
        fut_hpsm = ex.submit(timed, estimate_f0_hps_multi_wrapper, signal, sr)
        (res_hps, t_hps) = fut_hps.result()
        (res_hpsm, t_hpsm) = fut_hpsm.result()

    hps_best = _pick_hps_multi_best(res_hpsm)

    f_hp, conf_hp = compute_f0_HP_v2(
        signal,
        sr,
        f0_seed=f0_seed,
        window_factor=4,
        max_shift_cents=25.0,
    )
    f0_refined = f_hp if f_hp is not None else f0_seed

    decision = PrecisionRefineDecision(
        f0_seed=f0_seed,
        f0_refined=f0_refined,
        hps_result=res_hps,
        hps_multi_result=res_hpsm,
        hps_best=hps_best,
        hp_confidence=conf_hp,
        method="hp_v2_from_seed",
    )

    timings = {
        "hps_ms": float(t_hps),
        "hps_multi_ms": float(t_hpsm),
    }
    return decision, timings


# ──────────────────────────────────────────────
# MAIN
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
    use_informed_expected: bool = True,
    use_uninformed_partitioned: bool = False,
    yin_partition_mode: str = "binary_tree",
    yin_target_width_oct: float = 1.0,
    yin_max_depth: int = 5,
    context_expected_weight: float = 0.0,
    context_recent_weight: float = 0.0,
    recent_notes_hz: list[float] | None = None,
    context_cents_sigma: float = 50.0,
) -> NoteAnalysisResult:

    midi_expected = freq_to_midi(expected_freq)
    print(f"\n🎹  {bold(note_name)} expected → MIDI {midi_expected} | {expected_freq:.2f} Hz")

    start_all = time.perf_counter()

    # 1) PRIMARY SEED STRATEGY
    seed_decision, seed_timings = _detect_primary_seed(
        signal=signal,
        sr=sr,
        expected_freq=expected_freq,
        use_informed_expected=use_informed_expected,
        use_uninformed_partitioned=use_uninformed_partitioned,
        yin_partition_mode=yin_partition_mode,
        yin_target_width_oct=yin_target_width_oct,
        yin_max_depth=yin_max_depth,
    )

    print(cyan(
        "⏱ seed timings → "
        + (f"informed={seed_timings['informed_ms']:.1f} ms | " if use_informed_expected else "")
        + (f"yin_part={seed_timings['partitioned_ms']:.1f} ms | " if use_uninformed_partitioned else "")
        + f"seed_total={seed_timings['seed_block_ms']:.1f} ms"
    ))

    if use_informed_expected:
        informed_f0 = getattr(seed_decision.informed_guess, "f0", None)
        print(gray(
            "🎯 informed seed → "
            f"f0={fmt_val(informed_f0)} | "
            f"conf={seed_decision.informed_confidence:.3f} | "
            f"Δexpected={fmt_val(seed_decision.informed_delta_cents)}c | "
            f"accepted={seed_decision.informed_accepted}"
            + (f" | reject_reason={seed_decision.informed_reject_reason}" if seed_decision.informed_reject_reason else "")
        ))

    if use_uninformed_partitioned:
        print(gray(
            "🎯 partitioned seed → "
            f"f0={fmt_val(seed_decision.partitioned_f0)} | "
            f"score={seed_decision.partitioned_score:.3f} | "
            f"history={len(seed_decision.partitioned_history)}"
        ))

    if seed_decision.selected is None:
        print(red("❌ Aucun seed primaire valide."))
        return NoteAnalysisResult(
            midi=int(round(midi_expected)),
            note_name=freq_to_note(expected_freq),
            valid=False,
            f0=None,
            confidence=0.0,
            expected_freq=expected_freq,
        )

    f0_seed = float(seed_decision.selected.f0)
    seed_confidence = float(seed_decision.selected.confidence)
    seed_method = seed_decision.selected.method

    dev_seed_cents = _cents_delta(f0_seed, expected_freq)
    seed_note_name = freq_to_note(f0_seed)

    print(green(
        f"🌱 primary seed {seed_method} → {f0_seed:.3f} Hz | guessed={seed_note_name} | Δexpected={dev_seed_cents:.1f}c"
    ))

    # 2) PRECISION REFINEMENT (HPS for analysis, HP_v2 for final refinement)
    refine_decision, refine_timings = _refine_precision_from_seed(
        signal=signal,
        sr=sr,
        f0_seed=f0_seed,
    )

    hps_best = refine_decision.hps_best
    if hps_best and hps_best.get("f0", 0) > 0:
        print(gray(f"[HPS-multi → best {hps_best['f0']:.2f} Hz]"))

    print(cyan(
        "⏱ refine timings → "
        f"HPS={refine_timings['hps_ms']:.1f} ms | "
        f"multi={refine_timings['hps_multi_ms']:.1f} ms"
    ))

    f0_ref = float(refine_decision.f0_refined)
    print(gray(f"🔧 HP refine → seed={f0_seed:.3f} | refined={f0_ref:.3f}"))

    refined_note_name = freq_to_note(f0_ref)
    refined_midi = int(round(freq_to_midi(f0_ref)))
    dev_cents = _cents_delta(f0_ref, expected_freq)

    valid = (
        f0_ref > 20
        and seed_confidence >= 0.25
    )

    # 3) ACOUSTIC ANALYSIS ON REFINED F0
    unison = analyze_unison(signal, sr, f0_ref, refined_midi, piano_type, era)
    print(gray(
        f"🎛 unison: n={getattr(unison, 'detected_n_components', '?')} "
        f"beat={fmt_val(getattr(unison, 'beat_hz_estimate', '?'))}Hz "
        f"severity={fmt_val(getattr(unison, 'severity', '?'))}"
    ))

    harmonics, partials, inharm = compute_partials_fft_peaks(
        signal, sr, f0_ref, nb_partials=8, search_width_cents=60, pad_factor=2
    )
    part_freqs = [float(p[0]) for p in partials]

    B_final = estimate_B(f0_ref, part_freqs) if compute_inharm else None
    avg_inh = compute_inharmonicity_avg(inharm) if compute_inharm else None
    print(gray(f"inharm avg={fmt_val(avg_inh)}  B={fmt_val(B_final)}"))

    spec = np.abs(np.fft.rfft(signal))
    spec /= (np.max(spec) or 1)
    fingerprint = spec[:512]

    raw_fp, norm_fp = harmonic_spectrum_fft(signal, sr, f0_ref, nb_harmonics=8)

    response_data = None
    if compute_response_data and (len(signal) / sr) >= 3:
        response_data = compute_response(signal, sr, f0_ref)
        print(gray("📈 response curve computed"))

    t_total = (time.perf_counter() - start_all) * 1000
    print(green(
        f"✅ DONE — guessed={refined_note_name} | f0={f0_ref:.3f} Hz | conf={seed_confidence:.2f} | "
        f"B={fmt_val(B_final)} | valid={valid} | total={t_total:.1f} ms"
    ))

    # 4) DEBUG / SERIALIZATION
    sub_informed = _extract_subresults(seed_decision.informed_guess)
    sub_partitioned = None
    if seed_decision.partitioned_f0:
        sub_partitioned = {
            "yin_partitioned": {
                "f0": float(seed_decision.partitioned_f0),
                "conf": float(seed_decision.partitioned_score),
                "score": float(seed_decision.partitioned_score),
            }
        }

    sub_hps = _extract_subresults(refine_decision.hps_result)
    sub_hpsm = None
    if refine_decision.hps_best:
        sub_hpsm = {
            "hps_multi": {
                "f0": float(refine_decision.hps_best.get("f0", 0)),
                "conf": float(refine_decision.hps_best.get("conf", refine_decision.hps_best.get("quality", 0))),
                "score": float(refine_decision.hps_best.get("score", 0)),
            }
        }

    sub_all = {}
    for s in (sub_partitioned, sub_informed, sub_hps, sub_hpsm):
        if s:
            sub_all.update(s)

    guessed_note = GuessNoteResult(
        f0=float(f0_ref),
        confidence=float(seed_confidence),
        method=str(seed_method),
        envelope_band=getattr(seed_decision.informed_guess, "envelope_band", "unknown"),
        midi=refined_midi,
        note_name=refined_note_name,
        subresults=sub_all,
        seed={
            "f0": f0_seed,
            "method": seed_method,
            "confidence": seed_confidence,
            "fallback_used": seed_decision.fallback_used,
            "informed_accepted": seed_decision.informed_accepted,
            "informed_reject_reason": seed_decision.informed_reject_reason,
        },
    )

    guesses: dict[str, Any] = {
        "yin_partitioned": {
            "f0": float(seed_decision.partitioned_f0) if seed_decision.partitioned_f0 else None,
            "confidence": float(seed_decision.partitioned_score) if use_uninformed_partitioned else None,
            "method": "yin_partitioned",
            "subresults": sub_partitioned,
        },
        "informed_expected": {
            "f0": getattr(seed_decision.informed_guess, "f0", 0.0),
            "confidence": float(seed_decision.informed_confidence),
            "method": "informed_expected",
            "accepted": seed_decision.informed_accepted,
            "delta_expected_cents": seed_decision.informed_delta_cents,
            "reject_reason": seed_decision.informed_reject_reason,
            "subresults": sub_informed,
        },
        "hps": {
            "f0": refine_decision.hps_result.get("f0", 0.0) if isinstance(refine_decision.hps_result, dict) else None,
            "confidence": refine_decision.hps_result.get("quality", 0.0) if isinstance(refine_decision.hps_result, dict) else None,
            "method": "HPS",
            "subresults": sub_hps,
        },
        "hps_multi": {
            "f0": refine_decision.hps_best.get("f0", 0.0) if refine_decision.hps_best else 0.0,
            "confidence": refine_decision.hps_best.get("conf", refine_decision.hps_best.get("quality", 0.0)) if refine_decision.hps_best else 0.0,
            "method": "HPS-multi",
            "subresults": sub_hpsm,
        },
    }

    hps_payload: dict[str, Any] = {
        "f0": refine_decision.hps_result.get("f0") if isinstance(refine_decision.hps_result, dict) else None,
        "quality": refine_decision.hps_result.get("quality") if isinstance(refine_decision.hps_result, dict) else None,
        "B": refine_decision.hps_result.get("B") if isinstance(refine_decision.hps_result, dict) else None,
        "method": "HPS-wrapper",
    }
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

        append_diagnosis_row({
            "timestamp": timestamp,
            "expected_note": freq_to_note(expected_freq),
            "expected_freq": fmt(expected_freq),
            "f0_seed_note": freq_to_note(f0_seed),
            "yinp_mode": yin_partition_mode if use_uninformed_partitioned else "",
            "yinp_f0": fmt(seed_decision.partitioned_f0) if (use_uninformed_partitioned and seed_decision.partitioned_f0) else "",
            "yinp_score": fmt(seed_decision.partitioned_score) if use_uninformed_partitioned else "",
            "time_yinp_ms": fmt(seed_timings["partitioned_ms"], 2) if use_uninformed_partitioned else "",
            "essentia_f0": fmt(getattr(seed_decision.informed_guess, "f0", 0.0)),
            "essentia_conf": fmt(seed_decision.informed_confidence),
            "hps_f0": fmt(refine_decision.hps_result.get("f0", 0.0) if isinstance(refine_decision.hps_result, dict) else 0.0),
            "hps_q": fmt(refine_decision.hps_result.get("quality", 0.0) if isinstance(refine_decision.hps_result, dict) else 0.0),
            "hpsm_f0": fmt(refine_decision.hps_best.get("f0", 0.0) if refine_decision.hps_best else 0.0),
            "hpsm_q": fmt(refine_decision.hps_best.get("conf", refine_decision.hps_best.get("quality", 0.0)) if refine_decision.hps_best else 0.0),
            "final_f0": fmt(f0_ref),
            "final_conf": fmt(seed_confidence),
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
        midi=refined_midi,
        note_name=refined_note_name,
        valid=bool(valid),
        f0=float(f0_ref),
        confidence=float(seed_confidence),
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
        hps=hps_payload,
        unison=safe_asdict(unison),
        response=response_data,
        time_essentia_ms=seed_timings["informed_ms"],
        time_pfd_ms=refine_timings["hps_ms"],
        time_parallel_ms=t_total,
    )