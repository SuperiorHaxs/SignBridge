"""
framing_diag.py
---------------
Camera/viewpoint normalization diagnostic for Practice mode.

Compares the FRAMING of a user's extracted pose against the training-data
distribution to answer "how well is this camera/viewpoint normalized?" -- a
different question than "is the sign correct." Framing is measured on the RAW
(pre-shoulder-normalization) 83-point keypoints, because shoulder-centering +
shoulder-scaling deliberately cancels out exactly the camera geometry we want
to see (scale, vertical position, camera pitch, torso framing, aspect).

Two pieces:
  * frame_stats(pose)      -> per-clip framing statistics (device-agnostic body
                              framing, not sign-specific).
  * framing_fit(stats, baseline) -> z-scores + a 0-100 "camera fit" score +
                              human-readable, directional issues.

The baseline (mean/std of each stat over the training pool) is produced offline
by compute_framing_baseline.py and loaded via load_baseline().

Self-contained: numpy + stdlib only, so both the app and the offline script use it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 83-pt layout: body 0-32, left hand 33-53, right hand 54-74, face 75-82.
NOSE, L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 15, 16, 23, 24
L_HAND_C, R_HAND_C = 42, 63   # middle-finger MCP (33+9 / 54+9)

BASELINE_PATH = Path(__file__).resolve().parent / "framing_baseline.json"

# Stats scored against the baseline. ONLY normalization-invariant ratios: the
# training pool is stored shoulder-centered + shoulder-scaled (absolute scale and
# position are gone, and the model normalizes them away at inference too), so
# absolute framing is neither measurable nor relevant. These ratios survive that
# normalization and capture what actually varies across cameras: pitch, lens
# foreshortening / aspect, and where the signing space sits relative to the body.
_NUMERIC = ["pitch_ratio", "body_aspect", "nose_rel_y",
            "hand_rest_y_left", "hand_rest_y_right"]

# Framing completeness (fraction of frames with the hips visible). NOT affine
# invariant, but it matters: it catches "I'm zoomed in / my torso is cut off"
# -- the case the invariant ratios above miss (and can't even be computed when
# the hips leave frame). Handled by a dedicated rule in framing_fit.
_COVERAGE_KEY = "hips_visible_frac"
_BASELINE_KEYS = _NUMERIC + [_COVERAGE_KEY]   # stats persisted in the baseline

_Z_WARN, _Z_ERROR = 2.0, 3.0


def _valid(pt) -> bool:
    """A keypoint is present if it isn't the all-zero filler. Pool pickles are
    (x,y,z) with no visibility channel; live extraction adds a 4th visibility
    value -- honor it when present."""
    nonzero = not (float(pt[0]) == 0.0 and float(pt[1]) == 0.0)
    if len(pt) >= 4:
        return float(pt[3]) > 0.3 and nonzero
    return nonzero


def _path_len(pts) -> float:
    if len(pts) < 2:
        return 0.0
    p = np.asarray(pts, dtype=np.float32)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def frame_stats(pose) -> dict:
    """Per-clip framing statistics from raw (T,83,4) image-normalized keypoints.
    Returns {} if the body isn't reliably detected."""
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[1] < 33:
        return {}

    sw, msy, tl, pr, nr, sb = [], [], [], [], [], []
    lh_ry, rh_ry = [], []
    lw, rw = [], []
    hip_ok = 0

    for f in pose:
        lsh, rsh, nose = f[L_SH], f[R_SH], f[NOSE]
        if not (_valid(lsh) and _valid(rsh)):
            continue
        ms_y = (lsh[1] + rsh[1]) / 2.0
        shoulder_w = float(np.hypot(lsh[0] - rsh[0], lsh[1] - rsh[1]))
        if shoulder_w < 1e-4:
            continue
        sw.append(shoulder_w)
        msy.append(float(ms_y))
        # Room below the shoulders for the signing space, in shoulder-widths.
        # Frames here are the 4:3 sent frame (bottom edge = 1.0). Small = too close.
        sb.append(float((1.0 - ms_y) / shoulder_w))
        if _valid(nose):
            nr.append(float((ms_y - nose[1]) / shoulder_w))  # head-above-shoulders, scale-free

        lhip, rhip = f[L_HIP], f[R_HIP]
        if _valid(lhip) and _valid(rhip):
            hip_ok += 1
            mhy = (lhip[1] + rhip[1]) / 2.0
            torso = float(mhy - ms_y)
            if torso > 1e-4:
                tl.append(torso)
                if _valid(nose):
                    pr.append(float((ms_y - nose[1]) / torso))  # camera-pitch proxy

        lhc, rhc = f[L_HAND_C], f[R_HAND_C]
        if _valid(lhc):
            lh_ry.append(float((lhc[1] - ms_y) / shoulder_w))
        if _valid(rhc):
            rh_ry.append(float((rhc[1] - ms_y) / shoulder_w))
        lwr, rwr = f[L_WR], f[R_WR]
        if _valid(lwr):
            lw.append([float(lwr[0]), float(lwr[1])])
        if _valid(rwr):
            rw.append([float(rwr[0]), float(rwr[1])])

    if len(sw) < 3:
        return {}

    def med(a):
        return float(np.median(a)) if len(a) else None

    stats = {
        "shoulder_width":    med(sw),      # ~1 in the normalized pool; kept for debug only
        "midshoulder_y":     med(msy),     # ~0 in the normalized pool; debug only
        "torso_len":         med(tl),
        "hips_visible_frac": round(hip_ok / len(sw), 3),
        "pitch_ratio":       med(pr),
        "nose_rel_y":        med(nr),
        "space_below":       med(sb),
        "hand_rest_y_left":  med(lh_ry),
        "hand_rest_y_right": med(rh_ry),
        "frames_used":       len(sw),
    }
    if stats["torso_len"]:
        stats["body_aspect"] = stats["shoulder_width"] / stats["torso_len"]
    lp, rp = _path_len(lw), _path_len(rw)
    stats["motion_lr_ratio"] = float((lp + 1e-6) / (rp + 1e-6))
    stats["dominant_hand"] = ("left" if lp > rp * 1.3 else
                              "right" if rp > lp * 1.3 else "both")
    return stats


# Directional, human-readable messages keyed by (stat, sign-of-z).
def _msg(key: str, z: float) -> str:
    hi = z > 0
    return {
        "pitch_ratio":  "Camera angle looks off (not eye-level) — reference clips are filmed head-on.",
        "body_aspect":  "Your body proportions look stretched/squashed vs training (possible aspect-ratio or lens issue).",
        "nose_rel_y":   "Your head-to-shoulders proportion differs from training — likely camera tilt or height.",
        "hand_rest_y_left":  ("Your left hand sits lower in the signing space than in training."
                              if hi else "Your left hand sits higher in the signing space than in training."),
        "hand_rest_y_right": ("Your right hand sits lower in the signing space than in training."
                              if hi else "Your right hand sits higher in the signing space than in training."),
    }.get(key, f"{key} differs from training ({z:+.1f}σ).")


def framing_fit(stats: dict, baseline: dict, portrait: bool = False) -> dict:
    """Score a clip's framing vs the baseline. Returns
    {score, issues[], severity[], z{}, stats}.

    portrait: set when the source camera is portrait (phone). The client crops
    every device to a fixed 4:3 window upstream, so a portrait source loses its
    lower torso by construction -- coverage is then expected-low, not a user
    error, so we nudge gently instead of penalizing."""
    if not stats:
        return {"score": None,
                "issues": ["Body not clearly detected — center yourself in frame."],
                "severity": ["error"], "z": {}, "stats": {}}
    if not baseline:
        return {"score": None, "issues": ["No framing baseline available yet."],
                "severity": ["info"], "z": {}, "stats": stats}

    # Compute the baseline-relative z-scores for LOGGING/analysis only. They are
    # NOT surfaced as user warnings: the WLASL baseline's viewpoint stds are far
    # too tight to fairly judge a real camera (aspect/anatomy differ), so gating
    # on them just nags. Handedness/mirroring is likewise dropped -- it false-flags
    # any left-dominant or two-handed sign (frames are sent un-mirrored anyway).
    z = {}
    for k in _NUMERIC:
        b = baseline.get(k)
        v = stats.get(k)
        if not b or v is None or b.get("std", 0) < 1e-6:
            continue
        z[k] = round(float((v - b["mean"]) / b["std"]), 2)

    # User-facing verdict = geometric distance only (robust, actionable, matches
    # the live border): room below the shoulders for the signing space.
    issues, severity, level = [], [], "good"
    sb = stats.get("space_below")
    if sb is not None:
        if sb < 0.45:
            issues.append("Too close — move back so your hands have room below your shoulders.")
            severity.append("warn"); level = "bad"
        elif sb < 0.75:
            issues.append("A little close — move back slightly.")
            severity.append("info"); level = "warn"
    if not issues:
        issues = ["Framing looks good."]
        severity = ["ok"]
    score = 60 if level == "bad" else (85 if level == "warn" else 96)
    return {"score": score, "issues": issues, "severity": severity, "z": z, "stats": stats}


_BASELINE_CACHE = {"loaded": False, "data": None}


def load_baseline() -> dict | None:
    """Load the framing baseline JSON once (cached). None if not generated yet."""
    if _BASELINE_CACHE["loaded"]:
        return _BASELINE_CACHE["data"]
    _BASELINE_CACHE["loaded"] = True
    try:
        _BASELINE_CACHE["data"] = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _BASELINE_CACHE["data"] = None
    return _BASELINE_CACHE["data"]
