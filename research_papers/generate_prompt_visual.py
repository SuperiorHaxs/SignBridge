#!/usr/bin/env python3
"""
Generate a tight, poster-ready visual summarizing the two-stage LLM prompt
architecture for ASL-to-English sentence construction.

Output: research_papers/prompt_visual.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os


def main():
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # === TITLE ===
    ax.text(5.5, 8.2, 'LLM Prompt Architecture for ASL-to-English Translation',
            ha='center', fontsize=16, fontweight='bold', color='#1B5E20')
    ax.text(5.5, 7.9, 'Two-stage pipeline: Stage 1 selects glosses for coherence, Stage 2 constructs the sentence',
            ha='center', fontsize=10, color='#555')

    # =========================================================================
    # INPUT BOX (top left)
    # =========================================================================
    inp = FancyBboxPatch((0.3, 7.0), 4.9, 0.75,
                          boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', lw=2)
    ax.add_patch(inp)
    ax.text(0.5, 7.58, 'MODEL INPUT', fontsize=11, fontweight='bold', color='#1565C0')
    ax.text(0.5, 7.35, "Pos 1: 'give' (85%),  'want' (8%),  'take' (4%)\n"
                        "Pos 2: 'swimming' (40%),  'book' (32%),  'apple' (22%)\n"
                        "Pos 3: 'friend' (70%),  'man' (20%),  'son' (8%)",
            fontsize=8.5, fontfamily='monospace', color='#333', va='top', linespacing=1.3)

    # Arrow down
    ax.annotate('', xy=(2.75, 6.9), xytext=(2.75, 7.0),
                arrowprops=dict(arrowstyle='->', color='#999', lw=2))

    # =========================================================================
    # STAGE 1 BOX
    # =========================================================================
    s1 = FancyBboxPatch((0.3, 4.85), 4.9, 2.0,
                         boxstyle="round,pad=0.1", facecolor='#FFF3E0', edgecolor='#E65100', lw=2)
    ax.add_patch(s1)
    ax.text(2.75, 6.68, 'STAGE 1: Gloss Selection', ha='center', fontsize=12,
            fontweight='bold', color='#E65100')

    rules_s1 = (
        "1. Semantic Coherence over Confidence\n"
        "    mother+help+child > mother+help+deaf\n\n"
        "2. Object Compatibility\n"
        "    give + book (object)  not  give + swimming (activity)\n\n"
        "3. Temporal Pairing\n"
        "    'before' pairs with time nouns: year, day, week\n"
        "    'year' at 26% beats 'help' at 51% when 'before' present\n\n"
        "4. No Duplicate Words across positions\n"
        "    Keep word at highest-confidence position only"
    )
    ax.text(0.55, 6.42, rules_s1, fontsize=9, va='top', linespacing=1.25,
            color='#333', fontfamily='monospace')

    # Arrow down from Stage 1
    ax.annotate('', xy=(2.75, 4.75), xytext=(2.75, 4.85),
                arrowprops=dict(arrowstyle='->', color='#999', lw=2))

    # Selection result
    sel = FancyBboxPatch((0.6, 4.2), 4.3, 0.50,
                          boxstyle="round,pad=0.08", facecolor='#C8E6C9', edgecolor='#2E7D32', lw=1.5)
    ax.add_patch(sel)
    ax.text(2.75, 4.53, 'Selected: ["give", "book", "friend"]', ha='center',
            fontsize=10, fontweight='bold', color='#2E7D32', fontfamily='monospace')
    ax.text(2.75, 4.30, "'book' chosen over 'swimming' \u2014 you can give objects, not activities",
            ha='center', fontsize=8.5, color='#555', fontstyle='italic')

    # Arrow down
    ax.annotate('', xy=(2.75, 4.1), xytext=(2.75, 4.2),
                arrowprops=dict(arrowstyle='->', color='#999', lw=2))

    # =========================================================================
    # STAGE 2 BOX
    # =========================================================================
    s2 = FancyBboxPatch((0.3, 2.15), 4.9, 1.90,
                         boxstyle="round,pad=0.1", facecolor='#E8F5E9', edgecolor='#2E7D32', lw=2)
    ax.add_patch(s2)
    ax.text(2.75, 3.88, 'STAGE 2: Sentence Construction', ha='center', fontsize=12,
            fontweight='bold', color='#2E7D32')

    rules_s2 = (
        "1. Arrange glosses in grammatical English order\n\n"
        "2. Add only fillers: articles, prepositions, pronouns\n"
        "    (the, a, my, to, in, with, is, are)\n\n"
        "3. Never add negations or meaning-altering words\n\n"
        "4. Add possessives for relational nouns\n"
        '    "friend" \u2192 "my friend",  "son" \u2192 "my son"\n\n'
        "5. Correct verb forms: enjoy + gerund, want + infinitive"
    )
    ax.text(0.55, 3.62, rules_s2, fontsize=9, va='top', linespacing=1.2,
            color='#333', fontfamily='monospace')

    # Arrow down
    ax.annotate('', xy=(2.75, 2.05), xytext=(2.75, 2.15),
                arrowprops=dict(arrowstyle='->', color='#999', lw=2))

    # Output
    out = FancyBboxPatch((0.6, 1.45), 4.3, 0.55,
                          boxstyle="round,pad=0.08", facecolor='#1B5E20', edgecolor='#1B5E20', lw=2)
    ax.add_patch(out)
    ax.text(2.75, 1.82, '"Give the book to my friend."', ha='center',
            fontsize=12, fontweight='bold', color='white', fontstyle='italic')
    ax.text(2.75, 1.57, 'used: [give, book, friend]  |  dropped: []',
            ha='center', fontsize=8.5, color='#A5D6A7', fontfamily='monospace')

    # =========================================================================
    # RIGHT SIDE: Key Semantic Rules
    # =========================================================================
    rules_box = FancyBboxPatch((5.6, 4.85), 5.1, 2.75,
                                boxstyle="round,pad=0.1", facecolor='#F3E5F5', edgecolor='#7B1FA2', lw=2)
    ax.add_patch(rules_box)
    ax.text(8.15, 7.40, 'Key Semantic Rules', ha='center', fontsize=12,
            fontweight='bold', color='#7B1FA2')

    rules_right = (
        "VERB vs NOUN at Position 1\n"
        '  Prefer nouns as subjects: "The man..." not "Help..."\n\n'
        "TEMPORAL WORD PAIRING\n"
        "  before/after/later \u2192 scan for time nouns (year, day)\n"
        '  "A year before" \u2713   "Help before" \u2717 (before what?)\n\n'
        "OBJECT COMPATIBILITY\n"
        "  give/take \u2192 needs objects: book, apple, jacket\n"
        "  drink/eat \u2192 needs consumables: water, food\n"
        '  enjoy/like \u2192 can take anything: bowling, apples\n\n'
        "PROPER NOUNS over GENERIC\n"
        '  "Africa is hot" > "The man is hot" (more specific)\n\n'
        "SENTENCE COMPLETENESS\n"
        '  "but" at pos 1 \u2192 incomplete, prefer noun/adj instead'
    )
    ax.text(5.85, 7.15, rules_right, fontsize=9, va='top', linespacing=1.25,
            color='#333', fontfamily='monospace')

    # =========================================================================
    # DO / DON'T example box (bottom right)
    # =========================================================================
    do_box = FancyBboxPatch((5.6, 2.25), 5.1, 2.35,
                             boxstyle="round,pad=0.1", facecolor='#FAFAFA', edgecolor='#666', lw=1.5)
    ax.add_patch(do_box)
    ax.text(8.15, 4.40, 'Example: Meaning Over Confidence', ha='center', fontsize=11,
            fontweight='bold', color='#333')

    ax.text(5.85, 4.12,
            "Pos 1: 'give' (85%)    Pos 2: 'swimming' (40%)    Pos 3: 'friend' (70%)\n"
            "                              'book' (32%)\n"
            "                              'apple' (22%)",
            fontsize=8.5, va='top', fontfamily='monospace', color='#555', linespacing=1.3)

    # DO
    do_rect = FancyBboxPatch((5.85, 3.05), 4.6, 0.50,
                              boxstyle="round,pad=0.06", facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1.5)
    ax.add_patch(do_rect)
    ax.text(5.95, 3.40, '\u2713', fontsize=14, color='#2E7D32', fontweight='bold', va='center')
    ax.text(6.25, 3.40, '"Give the book to my friend."', fontsize=10,
            color='#2E7D32', fontweight='bold', va='center')
    ax.text(6.25, 3.15, "'book' is an object you can give", fontsize=8.5,
            color='#555', fontstyle='italic', va='center')

    # DON'T
    dont_rect = FancyBboxPatch((5.85, 2.35), 4.6, 0.62,
                                boxstyle="round,pad=0.06", facecolor='#FFEBEE', edgecolor='#C62828', lw=1.5)
    ax.add_patch(dont_rect)
    ax.text(5.95, 2.80, '\u2717', fontsize=14, color='#C62828', fontweight='bold', va='center')
    ax.text(6.25, 2.80, '"Give swimming to my friend."', fontsize=10,
            color='#C62828', fontweight='bold', va='center')
    ax.text(6.25, 2.52, "'swimming' is an activity \u2014 violates object compatibility\n"
                         "even though it has higher confidence (40% > 32%)",
            fontsize=8.5, color='#555', fontstyle='italic', va='center', linespacing=1.3)

    # =========================================================================
    # Bottom: stats bar
    # =========================================================================
    stats = FancyBboxPatch((0.3, 0.3), 10.4, 0.9,
                            boxstyle="round,pad=0.08", facecolor='#263238', edgecolor='none')
    ax.add_patch(stats)
    ax.text(5.5, 0.85, 'Two-Stage Pipeline  |  211-line prompt  |  '
                        '9 semantic rules  |  9 worked examples  |  '
                        'Powered by Gemini API',
            ha='center', va='center', fontsize=10.5, color='#B0BEC5')
    ax.text(5.5, 0.50, 'Meaning over confidence \u2014 a sensible sentence always beats a high-confidence nonsensical one',
            ha='center', va='center', fontsize=10, color='#FFB74D', fontweight='bold')

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ['png', 'pdf']:
        path = os.path.join(out_dir, f'prompt_visual.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path}")
    plt.close()


if __name__ == '__main__':
    main()
