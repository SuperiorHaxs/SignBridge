"""
compute_framing_baseline.py
---------------------------
Offline: build framing_baseline.json for the Practice camera-fit diagnostic.

Samples original (non-augmented) clips from the augmented_pool, computes the
normalization-invariant framing stats (framing_diag.frame_stats), and stores
per-stat mean/std plus the training handedness fraction. The Practice diagnostic
loads this to z-score a user's clip against the training-data framing.

Run from anywhere:
    python project-utilities/camera-framing/compute_framing_baseline.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import framing_diag as fd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POOL = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle"
OUT = Path(__file__).resolve().parent / "framing_baseline.json"

PER_GLOSS = 3       # originals sampled per gloss dir
MAX_CLIPS = 1200    # cap total clips for speed; plenty for stable mean/std


def main():
    if not POOL.exists():
        sys.exit(f"ERROR: pool not found: {POOL}")
    gloss_dirs = sorted(d for d in POOL.iterdir() if d.is_dir())
    acc = {k: [] for k in fd._BASELINE_KEYS}
    right_dom = hand_total = used = 0

    for gd in gloss_dirs:
        if used >= MAX_CLIPS:
            break
        # Originals only (skip *_aug_* copies) for a clean framing baseline.
        pkls = [p for p in sorted(gd.glob("*.pkl")) if "_aug_" not in p.name][:PER_GLOSS]
        for p in pkls:
            if used >= MAX_CLIPS:
                break
            try:
                d = pickle.load(open(p, "rb"))
                st = fd.frame_stats(d["keypoints"])
            except Exception:
                continue
            if not st:
                continue
            used += 1
            for k in fd._BASELINE_KEYS:
                if st.get(k) is not None:
                    acc[k].append(st[k])
            dh = st.get("dominant_hand")
            if dh in ("left", "right"):
                hand_total += 1
                right_dom += (dh == "right")

    baseline = {}
    for k, vals in acc.items():
        if len(vals) >= 20:
            a = np.asarray(vals, dtype=np.float32)
            lo, hi = np.percentile(a, [2, 98])   # trim extremes (aug/detector outliers)
            a = a[(a >= lo) & (a <= hi)]
            baseline[k] = {"mean": round(float(np.mean(a)), 5),
                           "std": round(float(np.std(a)), 5), "n": int(len(a))}
    if hand_total:
        baseline["dominant_hand_right_frac"] = round(right_dom / hand_total, 3)
    baseline["_meta"] = {"clips_used": used, "glosses_scanned": len(gloss_dirs),
                         "source": "augmented_pool originals"}

    OUT.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} from {used} clips across {len(gloss_dirs)} gloss dirs.\n")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
