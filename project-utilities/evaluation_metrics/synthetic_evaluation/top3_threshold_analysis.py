#!/usr/bin/env python3
"""
Analyze: at what Top-3 confidence threshold does LLM stop adding
statistically significant improvement to CTQI v3 score?
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

import sys

# Allow passing a custom results dir as CLI arg
if len(sys.argv) > 1:
    DATA_DIR = Path(sys.argv[1])
else:
    DATA_DIR = Path(__file__).parent / "evaluation_results_gemini_t1_n53_v4"

with open(DATA_DIR / "evaluation_results.json") as f:
    data = json.load(f)


def compute_ctqi_v3(ga, cf1, plausibility):
    """CTQI v3 = (GA/100) * (CF1/100) * (0.5 + 0.5 * P/100 * GA/100) * 100"""
    return (ga / 100) * (cf1 / 100) * (0.5 + 0.5 * (plausibility / 100) * (ga / 100)) * 100


# Build per-entry metrics
rows = []
skipped = 0
for entry in data:
    eid = entry['entry_id']
    glosses = entry['glosses']
    preds = entry['predicted_glosses']

    # Skip failed entries
    if entry.get('status') == 'failed':
        skipped += 1
        continue

    # Top-1 confidences
    top1_confs = [p['confidence'] for p in preds]
    avg_top1_conf = np.mean(top1_confs) * 100

    # Top-3 accuracy: is the correct gloss in top-3?
    top3_hits = 0
    top3_max_confs = []
    for gloss, pred in zip(glosses, preds):
        top3_glosses = [t['gloss'].upper() for t in pred['top_k']]
        if gloss.upper() in top3_glosses:
            top3_hits += 1
            for t in pred['top_k']:
                if t['gloss'].upper() == gloss.upper():
                    top3_max_confs.append(t['confidence'])
                    break
        else:
            top3_max_confs.append(0)

    top3_acc = (top3_hits / len(glosses)) * 100
    avg_top3_conf_correct = np.mean(top3_max_confs) * 100

    # Compute CTQI v3 from raw components
    # Use pre-computed v3 if available, otherwise compute from GA, CF1, P
    if 'baseline_composite_v3' in entry:
        baseline_v3 = entry['baseline_composite_v3']
        model_v3 = entry['model_composite_v3']
    else:
        # Baseline: GA = gloss_accuracy, CF1 = baseline_coverage_f1, P = baseline_plausibility
        b_ga = entry.get('gloss_accuracy', 0) or 0
        b_cf1 = entry.get('baseline_coverage_f1', 0) or 0
        b_p = entry.get('baseline_plausibility', 0) or 0
        baseline_v3 = compute_ctqi_v3(b_ga, b_cf1, b_p)

        # Model: GA = effective_gloss_accuracy, CF1 = model_coverage_f1, P = model_plausibility
        m_ga = entry.get('effective_gloss_accuracy', 0) or 0
        m_cf1 = entry.get('model_coverage_f1', 0) or 0
        m_p = entry.get('model_plausibility', 0) or 0
        model_v3 = compute_ctqi_v3(m_ga, m_cf1, m_p)

    delta_v3 = model_v3 - baseline_v3

    rows.append({
        'id': eid,
        'n_glosses': len(glosses),
        'avg_top1_conf': avg_top1_conf,
        'top3_acc': top3_acc,
        'avg_top3_conf_correct': avg_top3_conf_correct,
        'baseline_v3': baseline_v3,
        'model_v3': model_v3,
        'delta_v3': delta_v3,
        'gloss_accuracy': entry.get('gloss_accuracy', 0) or 0,
        'effective_gloss_accuracy': entry.get('effective_gloss_accuracy', 0) or 0,
    })

if skipped:
    print(f"NOTE: Skipped {skipped} failed entries, analyzing {len(rows)}/{len(data)}")

rows.sort(key=lambda r: r['avg_top1_conf'])

# ── Per-entry table ──
print("=" * 100)
print("PER-ENTRY DATA (sorted by Avg Top-1 Confidence)")
print("=" * 100)
hdr = f"{'ID':>3} {'#G':>3} {'Top1Conf%':>10} {'Top3Acc%':>9} {'Top3ConfCorr%':>14} {'BaseV3':>8} {'ModelV3':>9} {'Delta':>8} {'GA%':>6} {'EGA%':>6}"
print(hdr)
print("-" * 100)
for r in rows:
    print(f"{r['id']:>3} {r['n_glosses']:>3} {r['avg_top1_conf']:>10.1f} {r['top3_acc']:>9.0f} "
          f"{r['avg_top3_conf_correct']:>14.1f} {r['baseline_v3']:>8.1f} {r['model_v3']:>9.1f} "
          f"{r['delta_v3']:>8.1f} {r['gloss_accuracy']:>6.0f} {r['effective_gloss_accuracy']:>6.0f}")

# ── Analysis 1: By Top-3 Accuracy Bucket ──
print()
print("=" * 100)
print("ANALYSIS 1: LLM IMPROVEMENT BY TOP-3 ACCURACY BUCKET")
print("=" * 100)

buckets = {}
for r in rows:
    acc = r['top3_acc']
    if acc < 50:
        bucket = '<50%'
    elif acc < 75:
        bucket = '50-74%'
    elif acc < 100:
        bucket = '75-99%'
    else:
        bucket = '100%'
    buckets.setdefault(bucket, []).append(r)

for bucket in ['<50%', '50-74%', '75-99%', '100%']:
    if bucket not in buckets:
        continue
    entries = buckets[bucket]
    deltas = [e['delta_v3'] for e in entries]
    baselines = [e['baseline_v3'] for e in entries]
    models = [e['model_v3'] for e in entries]
    n = len(entries)

    if n >= 2:
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
    else:
        t_stat, p_val = float('nan'), float('nan')

    sig = 'YES' if (not np.isnan(p_val) and p_val < 0.05) else 'NO'
    print(f"\nBucket: Top-3 Acc = {bucket} (n={n})")
    print(f"  Avg Baseline CTQI v3: {np.mean(baselines):.2f}")
    print(f"  Avg Model CTQI v3:    {np.mean(models):.2f}")
    print(f"  Avg Delta:            {np.mean(deltas):+.2f} (std={np.std(deltas, ddof=1 if n > 1 else 0):.2f})")
    print(f"  Range of deltas:      [{min(deltas):.1f}, {max(deltas):.1f}]")
    print(f"  One-sample t-test (H0: delta=0): t={t_stat:.3f}, p={p_val:.6f}")
    print(f"  Statistically significant (p<0.05): {sig}")

# ── Analysis 2: By Avg Top-1 Confidence Threshold ──
print()
print("=" * 100)
print("ANALYSIS 2: LLM IMPROVEMENT BY AVG TOP-1 CONFIDENCE THRESHOLD")
print("=" * 100)

thresholds = [30, 40, 50, 60, 70, 80, 90]
for thresh in thresholds:
    above = [r for r in rows if r['avg_top1_conf'] >= thresh]
    below = [r for r in rows if r['avg_top1_conf'] < thresh]

    if len(above) < 2:
        continue

    deltas_above = [r['delta_v3'] for r in above]
    t_stat, p_val = stats.ttest_1samp(deltas_above, 0)
    sig = 'YES' if p_val < 0.05 else 'NO'

    print(f"\nThreshold: Avg Top-1 Conf >= {thresh}% (n={len(above)})")
    print(f"  Avg Delta CTQI v3: {np.mean(deltas_above):+.2f} (std={np.std(deltas_above, ddof=1):.2f})")
    print(f"  t={t_stat:.3f}, p={p_val:.6f} -> Significant: {sig}")

    if len(below) >= 2:
        deltas_below = [r['delta_v3'] for r in below]
        t_b, p_b = stats.ttest_1samp(deltas_below, 0)
        sig_b = 'YES' if p_b < 0.05 else 'NO'
        print(f"  Below threshold (n={len(below)}): Avg Delta={np.mean(deltas_below):+.2f}, p={p_b:.6f} -> Significant: {sig_b}")

# ── Analysis 3: Within 100% Top-3 accuracy, by confidence level ──
print()
print("=" * 100)
print("ANALYSIS 3: WITHIN 100% TOP-3 ACCURACY -- BY CONFIDENCE LEVEL")
print("=" * 100)

perfect_top3 = [r for r in rows if r['top3_acc'] == 100]
perfect_top3.sort(key=lambda r: r['avg_top3_conf_correct'])

print(f"\nEntries with 100% Top-3 accuracy: {len(perfect_top3)}")
if perfect_top3:
    print(f"Confidence range: {perfect_top3[0]['avg_top3_conf_correct']:.1f}% to {perfect_top3[-1]['avg_top3_conf_correct']:.1f}%")

conf_bins = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 101)]
for lo, hi in conf_bins:
    group = [r for r in perfect_top3 if lo <= r['avg_top3_conf_correct'] < hi]
    if not group:
        continue
    deltas = [r['delta_v3'] for r in group]
    n = len(group)
    if n >= 2:
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
    else:
        t_stat, p_val = float('nan'), float('nan')

    sig = 'YES' if (not np.isnan(p_val) and p_val < 0.05) else 'NO'
    label = f"{hi}%" if hi <= 100 else "100%]"
    print(f"\n  Correct-gloss-in-Top3 confidence [{lo}%, {hi}%) -- n={n}")
    print(f"    Avg Baseline CTQI v3: {np.mean([r['baseline_v3'] for r in group]):.2f}")
    print(f"    Avg Model CTQI v3:    {np.mean([r['model_v3'] for r in group]):.2f}")
    print(f"    Avg Delta: {np.mean(deltas):+.2f} (std={np.std(deltas, ddof=1 if n > 1 else 0):.2f})")
    print(f"    t={t_stat:.3f}, p={p_val:.6f} -> Significant: {sig}")
    for r in group:
        print(f"      Entry {r['id']:>2}: conf={r['avg_top3_conf_correct']:.1f}%, delta={r['delta_v3']:+.1f}, "
              f"base={r['baseline_v3']:.1f}, model={r['model_v3']:.1f}")

# ── Correlation ──
print()
print("=" * 100)
print("CORRELATION ANALYSIS")
print("=" * 100)

all_confs = [r['avg_top3_conf_correct'] for r in rows]
all_deltas = [r['delta_v3'] for r in rows]
r_corr, p_corr = stats.pearsonr(all_confs, all_deltas)
print(f"\nCorrelation: Top-3 correct confidence vs Delta CTQI v3")
print(f"  Pearson r = {r_corr:.4f}, p = {p_corr:.6f}")

all_top1 = [r['avg_top1_conf'] for r in rows]
r_corr2, p_corr2 = stats.pearsonr(all_top1, all_deltas)
print(f"\nCorrelation: Avg Top-1 confidence vs Delta CTQI v3")
print(f"  Pearson r = {r_corr2:.4f}, p = {p_corr2:.6f}")

# ── Entries where LLM hurt or had no effect ──
print()
print("=" * 100)
print("EDGE CASES")
print("=" * 100)

hurt = [r for r in rows if r['delta_v3'] < 0]
print(f"\nEntries where LLM HURT (delta < 0): {len(hurt)}")
for r in hurt:
    print(f"  Entry {r['id']:>2}: top3acc={r['top3_acc']:.0f}%, top1conf={r['avg_top1_conf']:.1f}%, "
          f"top3conf_correct={r['avg_top3_conf_correct']:.1f}%, delta={r['delta_v3']:+.1f}")

neutral = [r for r in rows if r['delta_v3'] == 0]
print(f"\nEntries where LLM had NO effect (delta = 0): {len(neutral)}")
for r in neutral:
    print(f"  Entry {r['id']:>2}: top3acc={r['top3_acc']:.0f}%, top1conf={r['avg_top1_conf']:.1f}%, "
          f"base={r['baseline_v3']:.1f}, model={r['model_v3']:.1f}")

# ── Summary finding ──
print()
print("=" * 100)
print("SUMMARY: THRESHOLD FINDING")
print("=" * 100)

# Find the threshold where delta becomes non-significant using sliding cutoff
print("\nSliding cutoff: entries with avg_top1_conf >= X, is delta still significant?")
for x in range(30, 96, 5):
    subset = [r for r in rows if r['avg_top1_conf'] >= x]
    if len(subset) < 3:
        continue
    deltas = [r['delta_v3'] for r in subset]
    t_stat, p_val = stats.ttest_1samp(deltas, 0)
    avg_d = np.mean(deltas)
    sig = 'YES' if p_val < 0.05 else 'NO'
    print(f"  >= {x:>2}%: n={len(subset):>2}, avg_delta={avg_d:>+6.2f}, p={p_val:.6f}, significant={sig}")

print("\nSliding cutoff: entries with avg_top1_conf >= X, is delta still significant? (ABOVE threshold only)")
for x in range(30, 96, 5):
    subset = [r for r in rows if r['avg_top1_conf'] >= x and r['avg_top1_conf'] < x + 10]
    if len(subset) < 3:
        continue
    deltas = [r['delta_v3'] for r in subset]
    t_stat, p_val = stats.ttest_1samp(deltas, 0)
    avg_d = np.mean(deltas)
    sig = 'YES' if p_val < 0.05 else 'NO'
    print(f"  [{x}%-{x+10}%): n={len(subset):>2}, avg_delta={avg_d:>+6.2f}, p={p_val:.6f}, significant={sig}")
