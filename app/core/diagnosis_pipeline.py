import math
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from pytune_dsp.types.schemas import NoteAnalysisResult
from pytune_dsp.analysis.partials import compute_partials_fft_peaks
from pytune_dsp.analysis.inharmonicity import compute_inharmonicity_avg, estimate_B
from pytune_dsp.analysis.spectrum import harmonic_spectrum_fft
from pytune_dsp.analysis.response import compute_response
from pytune_dsp.utils.note_utils import freq_to_midi, freq_to_note, midi_to_note
from pytune_dsp.utils.serialize import safe_asdict
from pytune_dsp.analysis.f0_HP_v2 import compute_f0_HP_v2

# pitch detectors
from pytune_dsp.analysis.pitch_detection_librosa import guess_note_librosa
from pytune_dsp.analysis.pitch_detection_expected import guess_note_expected_essentia
from pytune_dsp.analysis.unison import analyze_unison
from pytune_dsp.analysis.f0_HP import compute_f0_HP
from pytune_dsp.analysis.hps_seq import (
    estimate_f0_hps_wrapper,
    estimate_f0_hps_multi_wrapper
)

# NEW: Yin partitionné (non informé)
from pytune_dsp.analysis.yin_partitioned import (
    detect_f0_seed_partitioned,
    BandResult,
)
from pytune_dsp.analysis.yin_backend_essentia_frames import yin_backend_essentia_frames

# ──────────────────────────────────────────────
# COLOR HELPERS
# ──────────────────────────────────────────────

def c(t, code): return f"\033[{code}m{t}\033[0m"
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

    # optional detectors
    use_essentia: bool = False,
    use_librosa: bool = False,
    
    # NEW: Yin partitionné non informé
    use_yin_partitioned: bool = False,
    yin_partition_mode: str = "binary_tree",    # "octaves_8" | "octaves_4x2" | "binary_tree"
    yin_target_width_oct: float = 1.0,          # largeur finale (~1 octave par défaut)
    yin_max_depth: int = 5,


    # context weights
    context_expected_weight: float = 0.0,
    context_recent_weight: float = 0.0,
    recent_notes_hz: list[float] | None = None,
    context_cents_sigma: float = 50.0,
) -> NoteAnalysisResult:

    midi_expected = freq_to_midi(expected_freq)
    print(f"\n🎹  {bold(note_name)} expected → MIDI {midi_expected} | {expected_freq:.2f} Hz")

    # timing helper
    def timed(func, *args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        return out, (time.perf_counter() - t0) * 1000

    start_all = time.perf_counter()

    # build worker count
    n_workers = 2 + int(use_librosa) + int(use_essentia)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        fut_lib = fut_ess = None

        # librosa
        if use_librosa:
            fut_lib = ex.submit(timed, guess_note_librosa, signal, sr, True)

        # HPS engines
        fut_hps  = ex.submit(timed, estimate_f0_hps_wrapper, signal, sr, "auto")
        fut_hpsm = ex.submit(timed, estimate_f0_hps_multi_wrapper, signal, sr)

        # Essentia (EXPECTED)
        if use_essentia:
            fut_ess = ex.submit(
                timed,
                guess_note_expected_essentia,
                signal,
                sr,
                expected_freq,
                300.0,    # ±300 cents window
                True
            )
        # NEW: Yin partitionné (non informé)
        if use_yin_partitioned:
            fut_yin = ex.submit(
                timed,
                detect_f0_seed_partitioned,
                signal,
                sr,
                yin_partition_mode,
                yin_backend_essentia_frames,
                yin_target_width_oct,
                yin_max_depth,
            )


        # resolve futures
        guess_lib, t_lib = (None, 0.0)
        if use_librosa:
            guess_lib, t_lib = fut_lib.result()

        (res_hps, t_hps)   = fut_hps.result()
        (res_hpsm, t_hpsm) = fut_hpsm.result()

        guess_ess, t_ess = (None, 0.0)
        if use_essentia:
            guess_ess, t_ess = fut_ess.result()

        # NEW: Yin partitionné
        yin_f0_seed = None
        yin_score = 0.0
        t_yin = 0.0

        if use_yin_partitioned and fut_yin is not None:
            (yin_out, t_yin) = fut_yin.result()  # yin_out = (f0_seed, history)
            if yin_out is not None:
                yin_f0_seed, yin_history = yin_out
                if yin_history:
                    # score = max score des bandes ayant un f0 valide
                    valid_bands = [br for br in yin_history if br.f0 is not None]
                    if valid_bands:
                        best_br = max(valid_bands, key=lambda br: br.score)
                        yin_score = float(best_br.score)

        t_total = (time.perf_counter() - start_all) * 1000

    # normalize HPS-multi
    hpsm_best = None
    if isinstance(res_hpsm, dict):
        if "best" in res_hpsm:
            hpsm_best = res_hpsm["best"]
        elif isinstance(res_hpsm.get("candidates"), list) and res_hpsm["candidates"]:
            hpsm_best = res_hpsm["candidates"][0]

    print(cyan(
        "⏱ timings → "
        + (f"librosa={t_lib:.1f} ms | " if use_librosa else "")
        + (f"essentia={t_ess:.1f} ms | " if use_essentia else "")
        + (f"yin_part={t_yin:.1f} ms | " if use_yin_partitioned else "")
        + f"HPS={t_hps:.1f} ms | multi={t_hpsm:.1f} ms | total={t_total:.1f} ms"
    ))

    # ──────────────────────────────────────────────
    # collect proposals
    # ──────────────────────────────────────────────

    class Guess:
        def __init__(self, f0, conf, method):
            self.f0 = float(f0)
            self.confidence = float(conf)
            self.method = method

    def _cents_delta(f, ref):
        if f <= 0 or ref <= 0: return 1e9
        return 1200 * math.log2(f / ref)

    def _gauss(x, sig):
        if sig <= 0:
            return 1.0
        return math.exp(-0.5 * (x / sig) ** 2)

    proposals = {}

    # librosa
    if use_librosa and guess_lib and hasattr(guess_lib, "f0") and guess_lib.f0 > 0:
        proposals["librosa"] = Guess(guess_lib.f0, getattr(guess_lib,"confidence",0), "librosa")

    # essentia expected
    if use_essentia and guess_ess and hasattr(guess_ess, "f0") and guess_ess.f0 > 0:
        proposals["essentia"] = Guess(guess_ess.f0, getattr(guess_ess,"confidence",0), "essentia")

    # HPS
    if isinstance(res_hps, dict) and res_hps.get("f0", 0) > 0:
        proposals["hps"] = Guess(res_hps["f0"], res_hps.get("quality",0), "hps")

    # HPS multi
    if hpsm_best and hpsm_best.get("f0", 0) > 0:
        proposals["hps_multi"] = Guess(
            hpsm_best["f0"],
            hpsm_best.get("conf", hpsm_best.get("quality",0)),
            "hps_multi"
        )
        print(gray(f"[HPS-multi → best {hpsm_best['f0']:.2f} Hz]"))

    if not proposals:
        print(red("❌ Aucun moteur n’a donné de F0 valide."))
        return NoteAnalysisResult(
            note_name=note_name,
            valid=False,
            f0=None,
            confidence=0.0,
            expected_freq=expected_freq
        )

    # recent reference
    recent_ref_hz = None
    if recent_notes_hz:
        arr = [float(f) for f in recent_notes_hz if f and f>0]
        if arr:
            arr.sort()
            m=len(arr)//2
            recent_ref_hz = arr[m] if len(arr)%2 else 0.5*(arr[m-1]+arr[m])

    # anti-octave weighting
    if "hps" in proposals:
        hps_f = proposals["hps"].f0
        for g in proposals.values():
            r = abs(math.log2(g.f0/hps_f))
            if 0.45 < r < 0.55:
                g.confidence *= 0.6
            if r < 0.03:
                g.confidence *= 1.1

    # contextual weights
    for g in proposals.values():
        if context_expected_weight>0 and expected_freq>0:
            dc = abs(_cents_delta(g.f0, expected_freq))
            g.confidence *= (1 + context_expected_weight * _gauss(dc, context_cents_sigma))

        if context_recent_weight>0 and recent_ref_hz:
            dc = abs(_cents_delta(g.f0, recent_ref_hz))
            g.confidence *= (1 + context_recent_weight * _gauss(dc, context_cents_sigma))

    # select best
    best = max(proposals.values(), key=lambda g: g.confidence)
    f0_est = best.f0
    dev_cents = 1200*math.log2(f0_est/expected_freq)

    valid = (
        f0_est>20
        and abs(dev_cents) < 100
        and best.confidence >= 0.25
    )

    print(green(f"🌟 selected {best.method} → {f0_est:.3f} Hz (Δ={dev_cents:.1f}c)"))

    # UNISON
    unison = analyze_unison(signal, sr, f0_est, int(round(midi_expected)), piano_type, era)

    print(gray(
        f"🎛 unison: n={getattr(unison,'detected_n_components','?')} "
        f"beat={fmt_val(getattr(unison,'beat_hz_estimate','?'))}Hz "
        f"severity={fmt_val(getattr(unison,'severity','?'))}"
    ))

    f0_ref = f0_est
    force_hp = True
    if force_hp or getattr(unison, "recommend_f0_hp","") == "global":
        f_hp, conf_hp = compute_f0_HP_v2(
            signal,
            sr,
            f0_seed=f0_est,
            window_factor=4,
            max_shift_cents=25.0,  # ou 20.0 si tu veux être encore plus strict
        )
        f0_ref = f_hp if f_hp is not None else f0_est
    else:
        print(gray("⚙ skip HP-refine"))

    # harmonics / inharmonicity
    harmonics, partials, inharm = compute_partials_fft_peaks(
        signal, sr, f0_ref, nb_partials=8, search_width_cents=60, pad_factor=2
    )
    part_freqs = [float(p[0]) for p in partials]

    B_final = estimate_B(f0_ref, part_freqs) if compute_inharm else None
    avg_inh = compute_inharmonicity_avg(inharm) if compute_inharm else None

    print(gray(f"inharm avg={fmt_val(avg_inh)}  B={fmt_val(B_final)}"))

    # fingerprint
    spec = np.abs(np.fft.rfft(signal))
    spec /= (np.max(spec) or 1)
    fingerprint = spec[:512]

    raw_fp, norm_fp = harmonic_spectrum_fft(signal, sr, f0_ref, nb_harmonics=8)

    # subresults collector
    def extract_subresults(engine):
        if not engine: return None
        out = {}

        if hasattr(engine, "subresults") and isinstance(engine.subresults, dict):
            for k, v in engine.subresults.items():
                if isinstance(v, dict):
                    out[k.lower()] = {
                        "f0": float(v.get("f0",0)),
                        "conf": float(v.get("conf", v.get("confidence",0))),
                        "score": float(v.get("score",0)),
                    }

        elif isinstance(engine, dict):
            out["hps"] = {
                "f0": float(engine.get("f0",0)),
                "conf": float(engine.get("quality",0)),
                "score": float(engine.get("score",0)),
            }

        return out or None

    sub_lib  = extract_subresults(guess_lib)
    sub_ess  = extract_subresults(guess_ess)
    sub_hps  = extract_subresults(res_hps)
    sub_hpsm = None

    if hpsm_best:
        sub_hpsm = {
            "hps_multi":{
                "f0": float(hpsm_best.get("f0",0)),
                "conf": float(hpsm_best.get("conf", hpsm_best.get("quality",0))),
                "score": float(hpsm_best.get("score",0)),
            }
        }

    sub_all = {}
    for s in (sub_lib, sub_ess, sub_hps, sub_hpsm):
        if s: sub_all.update(s)

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
        "librosa": {
            "f0": getattr(guess_lib,"f0",0.0),
            "confidence": getattr(guess_lib,"confidence",0.0),
            "method": "librosa",
            "subresults": sub_lib
        },
        "essentia": {
            "f0": getattr(guess_ess,"f0",0.0),
            "confidence": getattr(guess_ess,"confidence",0.0),
            "method": "essentia",
            "subresults": sub_ess
        },
        "hps": {
            "f0": res_hps.get("f0",0.0) if isinstance(res_hps,dict) else None,
            "confidence": res_hps.get("quality",0.0) if isinstance(res_hps,dict) else None,
            "method": "HPS",
            "subresults": sub_hps
        },
        "hps_multi": {
            "f0": hpsm_best.get("f0",0.0) if hpsm_best else 0.0,
            "confidence": hpsm_best.get("conf",hpsm_best.get("quality",0.0)) if hpsm_best else 0.0,
            "method": "HPS-multi",
            "subresults": sub_hpsm
        }
    }

    # prepare response
    response_data = None
    if compute_response_data and (len(signal)/sr)>=3:
        response_data = compute_response(signal, sr, f0_ref)
        print(gray("📈 response curve computed"))

    print(green(
        f"✅ DONE — f0={f0_ref:.3f} Hz | conf={best.confidence:.2f} | "
        f"B={fmt_val(B_final)} | valid={valid}"
    ))
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

        # Robust selector for HPS-multi (best or candidates)
        def pick_hpsm_best(d):
            if not isinstance(d, dict):
                return {}
            if "best" in d:
                return d["best"]
            cand = d.get("candidates")
            return cand[0] if isinstance(cand, list) and cand else {}

        hpsm_csv = pick_hpsm_best(res_hpsm)

        append_diagnosis_row({
            "timestamp": timestamp,

            # Expected reference
            "expected_note": freq_to_note(expected_freq),
            "expected_freq": fmt(expected_freq),

            # f0_seed (before HP refine)
            "f0_seed_note": freq_to_note(f0_est),

            # NEW: Yin partitionné (non informé)
            "yinp_mode": yin_partition_mode if use_yin_partitioned else "",
            "yinp_f0": fmt(yin_f0_seed) if (use_yin_partitioned and yin_f0_seed) else "",
            "yinp_score": fmt(yin_score) if use_yin_partitioned else "",
            "time_yinp_ms": fmt(t_yin, 2) if use_yin_partitioned else "",

            # Essentia expected (if enabled)
            "essentia_f0": fmt(getattr(guess_ess, "f0", 0.0)),
            "essentia_conf": fmt(getattr(guess_ess, "confidence", 0.0)),

            # HPS
            "hps_f0": fmt(res_hps.get("f0", 0.0) if isinstance(res_hps, dict) else 0.0),
            "hps_q": fmt(res_hps.get("quality", 0.0) if isinstance(res_hps, dict) else 0.0),

            # HPS-multi
            "hpsm_f0": fmt(hpsm_csv.get("f0", 0.0)),
            "hpsm_q":  fmt(hpsm_csv.get("conf", hpsm_csv.get("quality", 0.0))),

            # HP refined final value
            "final_f0": fmt(f0_ref),
            "final_conf": fmt(best.confidence),

            # Cents deviation from expected
            "delta_cents": fmt(dev_cents),

            # Inharmonicity
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
        note_name=midi_to_note(int(note_name)),
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
        hps={"f0":res_hps.get("f0") if isinstance(res_hps,dict) else None,
             "quality":res_hps.get("quality") if isinstance(res_hps,dict) else None,
             "B":res_hps.get("B") if isinstance(res_hps,dict) else None,
             "method":"HPS-wrapper"},

        unison=safe_asdict(unison),
        response=response_data,

        time_librosa_ms=t_lib,
        time_essentia_ms=t_ess,
        time_pfd_ms=t_hps,
        time_parallel_ms=t_total,
    )