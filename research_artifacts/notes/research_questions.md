# Research Questions & Framing

**Overarching Research Question:**
> *"To what extent do standard MT metrics capture semantic fidelity in ASL-to-English translation, and how does upstream recognition uncertainty propagate into downstream translation quality?"*

**Title:** Can Intended Meaning Be Preserved in ASL-to-English Translation Through Semantic Coherence?

---

## Honest Critique: Research vs. Engineering

**Where the original framing reads as engineering:**
- Experiments 2 and 3 as originally conceived are essentially parameter sweeps -- "find the threshold where X works" and "find the max vocabulary size." These answer *operational* questions (how to deploy a system) rather than *scientific* questions (why does something happen, what does it reveal about language/translation).
- CTQI risks looking like "I made a score because existing scores didn't give me the numbers I wanted" unless it's grounded in a falsifiable hypothesis about *what makes ASL-English translation fundamentally different* from other translation tasks.
- An arbitrary 80% target in experiment 3 is an easy target for judges -- "why 80%?"

**Where it has real research potential:**
- Experiment 1 is genuinely strong. The claim that BLEU/BERT fail for ASL-English is a **testable, publishable hypothesis** -- ASL has different syntax, no word-for-word correspondence, and heavy contextual dependency. This is linguistically interesting.
- The connection between sign recognition confidence and downstream semantic quality is unexplored territory.

**The key shift: don't optimize a system -- investigate a phenomenon.**

---

## Reframed Experiments

### Experiment 1: Metric Validity (strongest piece)

**Hypothesis:** BLEU and BERTScore systematically underestimate translation quality for ASL-to-English compared to human judgments, due to ASL's non-linear grammar.

**Method:** Generate translations from sign sequences, collect human quality ratings (5 raters on 53 sentences), compute BLEU/BERT, and show the **correlation gap**. Then introduce CTQI v3 and show it correlates better with human judgment.

**This is research because:** You're making a falsifiable claim about metric validity and providing evidence.

**Key Finding:** Pearson r with human ratings (n=53, 5 raters): BLEU=0.758, BERTScore=0.807, CTQI v3=0.943. CTQI v3 is the strongest predictor of human-perceived translation quality.

---

### Experiment 2: Error Propagation / Phase Transition (reframe the threshold question)

**Hypothesis:** There exists a phase transition in translation quality -- below a critical recognition confidence, LLM reconstruction cannot recover intended meaning; above it, recovery is reliable.

**Method:** Same data, but frame it as studying **error propagation** through a pipeline. You're not "finding the best threshold" -- you're characterizing how upstream noise affects downstream semantics. Phase transitions and error propagation are well-studied phenomena. You're contributing evidence of one in a new domain.

**This is research because:** You're characterizing how errors cascade through a pipeline, not tuning a parameter.

**Key Finding:** The LLM rescue effect is statistically significant (p<0.001) and broadly positive across accuracy bands, but the magnitude and pattern differ between datasets:
- n=53 (in-domain): Largest improvement in the 70-80% band (delta +32, p=1.3e-03)
- n=150 (cross-domain): Largest improvement in the 90-100% band (delta +6, p=7.4e-06)

This suggests the LLM rescue is most effective when recognition provides sufficient correct signal, but the threshold depends on whether sentences use in-vocabulary or out-of-vocabulary signs.

---

### Experiment 3: Domain Vocabulary Scaling (reframe vocabulary size)

**Hypothesis:** Domain-constrained vocabularies reduce the entropy of the recognition task, and translation quality degrades predictably as vocabulary size increases (following an information-theoretic relationship).

**Method:** Instead of "find the max," **model the curve**. Does quality degrade linearly? Logarithmically? Is there a cliff? Fit it and explain *why* the shape is what it is.

**This is research because:** You're characterizing a relationship, not just finding a number.

**Key Finding:** Expanding from 38-class (healthcare) to 77-class (general + healthcare) causes significant accuracy degradation for shared signs (e.g., DOCTOR drops from 91.8% to 1.6% Top-1). The percentage of signs meeting the 90% Top-3 threshold drops sharply as vocabulary grows, establishing a practical limit on domain size for reliable translation.

---

## Judge-Proofing Tips

1. **Lead with the linguistics, not the engineering.** Your poster/paper intro should talk about ASL grammar being fundamentally different from spoken English -- that's what makes this interesting, not the ML pipeline.
2. **Human evaluation is non-negotiable.** Even a small-scale human study (3 raters, 50 sentences) transforms this from "I ran some code" to "I validated against ground truth." Use Krippendorff's alpha or Cohen's kappa.
3. **Name the phenomenon, not the tool.** "Error propagation in sign language translation pipelines" sounds like research. "Building a better ASL translator" sounds like engineering.
4. **Drop the 80% threshold** or justify it empirically (e.g., "below this, human raters classified translations as unintelligible"). Arbitrary cutoffs are an easy target for judges.
