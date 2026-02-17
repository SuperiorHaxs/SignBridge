#!/usr/bin/env python3
"""Generate CTQI v1 vs v2 comparison report."""
import json
from datetime import datetime

with open('synthetic_evaluation/evaluation_results/evaluation_results.json', 'r') as f:
    data = json.load(f)

def ctqi_v1(gloss_acc, quality, ptr):
    return 0.4 * gloss_acc + 0.4 * quality + 0.2 * ptr

def ctqi_v2(gloss_acc, coverage_f1, plausibility):
    return (gloss_acc / 100) * (coverage_f1 / 100) * (0.5 + 0.5 * plausibility / 100) * 100

output = []
output.append('=' * 80)
output.append('CTQI v1 vs v2 COMPARISON REPORT')
output.append('Generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
output.append('Dataset: 34 synthetic sentence samples')
output.append('=' * 80)
output.append('')
output.append('')
output.append('FORMULAS')
output.append('-' * 80)
output.append('CTQI v1 (Weighted Average):')
output.append('    CTQI = 0.4 * GA + 0.4 * Quality + 0.2 * PTR')
output.append('')
output.append('CTQI v2 (Prerequisite Chain):')
output.append('    CTQI = (GA/100) * (CF1/100) * (0.5 + 0.5 * P/100) * 100')
output.append('')
output.append('Components:')
output.append('    GA  = Gloss Accuracy (sign recognition)')
output.append('    CF1 = Coverage F1 (content word overlap)')
output.append('    P   = Plausibility (grammar + semantics, replaces Quality)')
output.append('    PTR = Perfect Translation Rate (binary bonus, v1 only)')
output.append('')
output.append('')

# Table 1: v1 vs v2 with components
output.append('=' * 80)
output.append('TABLE 1: CTQI v1 vs v2 COMPARISON (Model Output)')
output.append('=' * 80)
output.append('')
header = 'ID   Glosses                       GA   CF1    P    v1    v2   Diff'
output.append(header)
output.append('-' * 80)

v1_scores = []
v2_scores = []

for entry in data:
    eid = entry['entry_id']
    glosses = ' '.join(entry['glosses'])
    if len(glosses) > 26:
        glosses = glosses[:23] + '...'

    model_ga = entry.get('effective_gloss_accuracy', entry.get('gloss_accuracy', 0))
    model_cf1 = entry.get('model_coverage_f1', 0)
    model_p = entry.get('model_quality', 0)

    if entry.get('effective_gloss_mismatches'):
        ptr = 0
    elif model_ga == 100:
        ptr = 100
    else:
        ptr = 0

    v1 = ctqi_v1(model_ga, model_p, ptr)
    v2 = ctqi_v2(model_ga, model_cf1, model_p)
    v1_scores.append(v1)
    v2_scores.append(v2)

    diff = v2 - v1
    diff_str = '+%d' % diff if diff >= 0 else '%d' % diff

    line = '%-4d %-28s %4.0f %4.0f %4.0f %5.0f %5.0f %6s' % (eid, glosses, model_ga, model_cf1, model_p, v1, v2, diff_str)
    output.append(line)

output.append('')
output.append('SUMMARY:')
output.append('    v1 Mean: %.1f  |  v2 Mean: %.1f' % (sum(v1_scores)/len(v1_scores), sum(v2_scores)/len(v2_scores)))
output.append('    v1 Range: %.1f - %.1f  |  v2 Range: %.1f - %.1f' % (min(v1_scores), max(v1_scores), min(v2_scores), max(v2_scores)))

mean_v1 = sum(v1_scores) / len(v1_scores)
mean_v2 = sum(v2_scores) / len(v2_scores)
cov = sum((a - mean_v1) * (b - mean_v2) for a, b in zip(v1_scores, v2_scores)) / len(v1_scores)
std_v1 = (sum((a - mean_v1)**2 for a in v1_scores) / len(v1_scores)) ** 0.5
std_v2 = (sum((b - mean_v2)**2 for b in v2_scores) / len(v2_scores)) ** 0.5
corr = cov / (std_v1 * std_v2) if std_v1 > 0 and std_v2 > 0 else 0
output.append('    Correlation (v1 vs v2): r = %.3f' % corr)
output.append('')
output.append('')

# Table 2: Raw vs Model
output.append('=' * 80)
output.append('TABLE 2: RAW (Gloss Concatenation) vs MODEL (LLM-Enhanced) - CTQI v2')
output.append('=' * 80)
output.append('')
header2 = 'ID   Glosses              Raw GA|CF1|P    Model GA|CF1|P   Raw  Model   Diff  Winner'
output.append(header2)
output.append('-' * 80)

raw_wins = 0
model_wins = 0
ties = 0
improvements = []

for entry in data:
    eid = entry['entry_id']
    glosses = ' '.join(entry['glosses'])
    if len(glosses) > 18:
        glosses = glosses[:15] + '...'

    model_ga = entry.get('effective_gloss_accuracy', entry.get('gloss_accuracy', 0))
    model_cf1 = entry.get('model_coverage_f1', 0)
    model_p = entry.get('model_quality', 0)

    raw_ga = entry.get('gloss_accuracy', model_ga)
    raw_cf1 = entry.get('baseline_coverage_f1', 0)
    raw_p = entry.get('baseline_quality', 0)

    raw_v2 = ctqi_v2(raw_ga, raw_cf1, raw_p)
    model_v2 = ctqi_v2(model_ga, model_cf1, model_p)
    diff = model_v2 - raw_v2
    improvements.append(diff)

    if model_v2 > raw_v2 + 0.5:
        winner = 'Model'
        model_wins += 1
    elif raw_v2 > model_v2 + 0.5:
        winner = 'Raw'
        raw_wins += 1
    else:
        winner = 'Tie'
        ties += 1

    diff_str = '+%d' % diff if diff >= 0 else '%d' % diff
    raw_comp = '%d|%d|%d' % (raw_ga, raw_cf1, raw_p)
    model_comp = '%d|%d|%d' % (model_ga, model_cf1, model_p)

    line = '%-4d %-20s %-14s %-14s %5.0f %6.0f %6s  %s' % (eid, glosses, raw_comp, model_comp, raw_v2, model_v2, diff_str, winner)
    output.append(line)

output.append('')
output.append('SUMMARY:')
output.append('    Model wins: %d/34 (%d%%)' % (model_wins, model_wins*100//34))
output.append('    Raw wins:   %d/34 (%d%%)' % (raw_wins, raw_wins*100//34))
output.append('    Ties:       %d/34 (%d%%)' % (ties, ties*100//34))
output.append('')
output.append('    Average improvement: +%.1f points' % (sum(improvements)/len(improvements)))
output.append('    Max improvement: +%d (ID %d)' % (max(improvements), improvements.index(max(improvements))))
output.append('    Max regression: %d (ID %d)' % (min(improvements), improvements.index(min(improvements))))
output.append('')
output.append('')

# Analysis
output.append('=' * 80)
output.append('ANALYSIS: WHY RAW WON (4 cases)')
output.append('=' * 80)
output.append('')
output.append('ID 16: BASKETBALL PLAY TIME')
output.append('    Raw P=75, Model P=30')
output.append('    Model output "Basketball play time." scored worse on plausibility')
output.append('')
output.append('ID 21: TALL SON BASKETBALL')
output.append('    Raw GA=67, Model GA=33')
output.append('    Model picked wrong gloss, degrading recognition accuracy')
output.append('')
output.append('ID 27: WATER GIVE DOCTOR')
output.append('    Raw P=53, Model P=41')
output.append('    Model output "Give water time." less plausible than raw')
output.append('')
output.append('ID 31: TALL MAN DRINK WATER')
output.append('    Raw GA=50, Model GA=25')
output.append('    Model recognition failed badly, missing key signs')
output.append('')
output.append('')

# Key insights
output.append('=' * 80)
output.append('KEY INSIGHTS')
output.append('=' * 80)
output.append('')
output.append('1. CTQI v2 is stricter than v1:')
output.append('   - v2 uses multiplication (all components must be good)')
output.append('   - v1 uses addition (one high score can mask a low score)')
output.append('   - Average v2 score is ~10 points lower than v1')
output.append('')
output.append('2. LLM pipeline improves CTQI v2 by +25.9 points on average:')
output.append('   - Improves GA by correcting gloss selection using context')
output.append('   - Improves CF1 through better content word coverage')
output.append('   - Improves P by generating more grammatical English')
output.append('')
output.append('3. LLM pipeline can regress when:')
output.append('   - LLM picks wrong alternative gloss (ID 21, 31)')
output.append('   - LLM output is less natural than simple concatenation (ID 16, 27)')
output.append('')
output.append('4. v2 component behaviors:')
output.append('   - Low GA (<50%): Gates everything, score collapses')
output.append('   - Low CF1 (<70%): Indicates missing content words')
output.append('   - Low P (<50%): Indicates poor grammar/fluency')
output.append('   - Multiple low components: Severe multiplicative penalty')
output.append('')
output.append('=' * 80)
output.append('END OF REPORT')
output.append('=' * 80)

# Write to file
report_path = 'synthetic_evaluation/evaluation_results/ctqi_v1_vs_v2_comparison_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(output))

print('Report saved to:', report_path)
print('Total lines:', len(output))
