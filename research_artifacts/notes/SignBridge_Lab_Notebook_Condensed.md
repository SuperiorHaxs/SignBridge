# SignBridge: Real-Time ASL Recognition System — Engineering Lab Notebook

**Student:** Kavin Kumar Padmanabhan
**School:** Eastside Catholic School, Sammamish, WA
**Category:** TECA-HIE (Technology Enhances the Arts — Human Information Exchange)
**Project Start Date:** August 1, 2025
**GitHub:** https://github.com/SuperiorHaxs/SignBridge (125 commits)
**Live Demo:** https://huggingface.co/spaces/SuperiorHaxs/SignBridge

---

## Table of Contents

**TOC Page 1**
- Page 1 — Problem Definition (Aug 1)
- Page 2 — Research Questions & Hypotheses (Aug 1)
- Page 3 — Engineering Goals & Expected Outcomes (Aug 1)
- Page 4 — Background Research: Technical Sources (Aug, ongoing)
- Page 5 — Background Research: Linguistic/Cultural + Key Insight (Aug, ongoing)
- Page 6 — Design Brainstorming & Architecture (Sep, ongoing)
- Page 7 — Constraints & Performance Criteria (Aug, ongoing)
- Pages 8–9 — Design Iteration Log (Aug–Feb, ongoing)

**TOC Page 2**
- Page 10 — [Phase 1] Literature survey & initial findings (Aug 1)
- Page 11 — ASL linguistics: grammar & spatial referencing (Aug 8)
- Page 12 — CCIR training: problem statement workshop (Aug 15)
- Page 13 — CCIR training: experimental design & methods (Aug 22)
- Page 14 — Research synthesis: benchmarks & project plan (Aug 29)
- Page 15 — [Phase 2] OpenHands-HD keypoint expansion, 83 pts (Sep)
- Page 16 — Data augmentation pipeline, 50x expansion (Sep)
- Page 17 — First prototypes: 20-class & 50-class models (Oct)
- Page 18 — LLM integration design: semantic coherence (Oct)
- Page 19 — CTQI evaluation framework design (Oct)
- Page 20 — Prototype refinement & pre-paper baselines (early Nov)
- Page 21 — [Phase 3] Begin writing CCIR research paper (Nov 8)
- Page 22 — Submit paper to CCIR (Nov 24)
- Page 23 — Selected for CCIR presentation (late Nov)
- Page 24 — Prepare presentation slides & speaker notes (Dec 1–5)
- Page 25 — Present at CCIR Winter Symposium (Dec 6)

**TOC Page 3**
- Page 26 — [Phase 4] Resume dev: full pipeline integration (Dec 7)
- Page 27 — Temporal segmentation for continuous signing (Dec)
- Page 28 — Pipeline testing & debugging (Dec)
- *(Dec 25–30: Winter vacation)*
- Page 29 — [Phase 5] Real-time webcam demo application (Dec 31)
- Page 30 — Train 100-class model: 80.97% Top-1 (Jan)
- Page 31 — CTQI evaluation: full statistical analysis (Jan)
- Page 32 — Streaming inference & smart buffering (Jan)
- Page 33 — Model optimization: dropout, LR, label smoothing (Jan)
- Page 34 — LLM prompt iteration & error analysis (Jan)
- Page 35 — Incorporate CCIR feedback (Jan)
- Page 36 — Final metrics: p-values, Cohen's d, ablation (late Jan)
- Page 36.5 — CTQI v2 remodel: prerequisite chain design (Jan 30)
- Page 37 — [Phase 6] Docker + HuggingFace deployment (Feb 1)
- Page 38 — Deployed app testing & verification (Feb 5)
- Page 38.5 — Human survey design & deployment (Feb 7)
- Page 38.7 — Survey analysis: CTQI v2 weakness discovery (Feb 9)
- Page 38.8 — CTQI v3 design & human validation (Feb 10)
- Page 39 — [Phase 7] Begin CSRSEF materials: poster, notebook (Feb 14)
- Page 40 — Demo rehearsal & presentation practice (Feb)
- Page 41 — Final preparations: day before CSRSEF (Mar)
- Page 42 — Code Repository Reference (printed)
- Pages 43–86 — reserved for future work

---

## Page 1 — Problem Definition

- ~70 million Deaf individuals worldwide rely on sign languages; fewer than 20% of digital content receives accurate captions
- ASL and English are distinct linguistic systems with fundamentally different grammars (topic-comment vs SVO, spatial referencing, non-manual markers)
- Existing technology either produces awkward word-for-word "gloss," requires expensive specialized hardware, or stops at isolated sign recognition
- **Core problem: preserving meaning across two grammatically incompatible languages in real time**

---

## Page 2 — Research Questions & Hypotheses

**Research Questions:**
- **RQ1:** Can tracking more body points (83 vs 27) improve recognition accuracy with limited training data?
- **RQ2:** Can an LLM improve translation by considering sentence-level meaning rather than evaluating signs independently?
- **RQ3:** Can multi-dimensional quality measurement (CTQI) provide better assessment than single metrics?

**Hypotheses:**
- **H1 (Enhanced Body Tracking):** 83 keypoints (OpenHands-HD) + 50x augmentation will improve word-level accuracy vs standard 27-point methods
- **H2 (AI-Powered Word Selection):** Gemini LLM selecting from Top-K candidates based on semantic coherence will produce higher quality translations than Top-1 confidence alone
- **H3 (Multi-Dimensional Quality):** CTQI combining lexical accuracy, meaning preservation, and grammatical correctness will reveal improvements no single metric captures

---

## Page 3 — Engineering Goals

1. Build sign recognition system tracking 83 body points including detailed finger positions
2. Integrate Gemini LLM for grammatically correct sentence construction from Top-K predictions
3. Develop CTQI — multi-dimensional quality scoring framework
4. Create a working real-time ASL-to-English translation prototype

**Expected Outcomes:**
- Recognition accuracy 80%+ among Top-3 predictions
- Grammatical quality significantly improved over word-by-word output
- CTQI validated as superior to single metrics for quality assessment
- Functional, deployed system demonstrating viability

---

## Page 4 — Background Research: Technical Sources

| Source | Key Takeaway | Reviewed |
|--------|-------------|----------|
| Li et al. (2020), WLASL Dataset | Largest ASL dataset: 2,000 signs; WLASL-100: 100 classes, ~3.4 samples/class | Aug 2025 |
| Selvaraj et al. (2022), OpenHands | Pose-based SLR framework; 71.57% Top-1 on WLASL-100 with 27 keypoints | Aug 2025 |
| MediaPipe Holistic (Google) | Real-time pose: 33 body + 21/hand + 468 face landmarks | Aug 2025 |
| I3D Baseline (WLASL) | Video-based: 65.89% Top-1; too slow for real-time | Aug 2025 |
| Multi-stream CNN (SOTA) | Video-based SOTA: 81.38% Top-1; not real-time | Aug 2025 |
| Google Gemini (gemini-2.0-flash) | Fast inference LLM for real-time contextual sign selection | Sep 2025 |
| Zhang et al. (2020), BERTScore | Semantic similarity via BERT embeddings | Sep 2025 |
| Papineni et al. (2002), BLEU | Lexical similarity metric for MT evaluation | Sep 2025 |

## Page 5 — Background Research: Linguistic / Cultural Sources

| Source | Key Takeaway | Reviewed |
|--------|-------------|----------|
| Valli & Lucas, *Linguistics of ASL* | ASL has its own syntax, morphology, phonology — not derived from English | Aug 2025 |
| Stokoe (1960), Sign Language Structure | Foundational proof that ASL is a true language | Aug 2025 |
| World Federation of the Deaf (2023) | Technology should support, not replace sign language | Aug 2025 |
| NAD Position on ASL Recognition Tech | Deaf community: technology should empower, not replace interpreters | Aug 2025 |

### Key Insight

- ASL uses topic-comment structure: `STORE IX-1 GO YESTERDAY` = "I went to the store yesterday"
- Direct word-for-word mapping is linguistically invalid
- **This grammatical gap is why LLM post-processing is essential — it's linguistic bridging, not just cleanup**

---

## Page 6 — Design Brainstorming & Architecture

**Approach Comparison:**
- ~~I3D video-based~~ — 65.89% accuracy but too slow for real-time communication
- ~~Multi-stream CNN SOTA~~ — 81.38% but extremely slow, impractical
- ~~OpenHands baseline (27 pts)~~ — Real-time but only 71.57%, misses finger detail
- **OpenHands-HD (83 pts) + LLM** — Real-time, rich features, grammar bridging; each stage serves communication

**Three-Component Architecture:**
```
Component 1: OpenHands-HD Sign Recognition
  (83 keypoints, 279-dim features, 50x augmentation, Transformer, Top-K output)
    ↓
Component 2: LLM Sentence Construction
  (Gemini 2.0 Flash, Top-K selection, semantic coherence, grammar correction)
    ↓
Component 3: CTQI Evaluation
  (BLEU, BERTScore, Quality Score, Composite CTQI)
```

- Rejected confidence-only selection because correct sign often in Top-3 but not Top-1, and ignoring context produces "word salad"

---

## Page 7 — Constraints & Performance Criteria

| Criterion | Target | Achieved | Why It Matters |
|-----------|--------|----------|----------------|
| Top-1 Accuracy | > 80% | **80.97%** | Reliable sign recognition foundation |
| Top-3 Accuracy | > 90% | **91.62%** | Good LLM candidate pool |
| CTQI Score | > 75 | **78.16** | Holistic communication quality |
| Quality Score | > 70 | **74.56** | Grammatical correctness |
| Coverage F1 | > 80% | **87.62%** | Semantic content preserved |
| Perfect Translation Rate | > 60% | **67.6%** | All glosses correct |
| BLEU Score | > 50 | **56.53** | Lexical similarity |
| BERTScore | > 95% | **96.30%** | Semantic similarity |
| Latency | < 2s | **< 2s** | Real-time communication |
| Hardware | Consumer webcam | **Yes** | Accessibility |
| Statistical significance | p < 0.05 | **p < 0.001** | Scientifically validated |

**Budget:** $0 hardware (existing laptop + webcam), free API tiers, open-source tools/datasets

---

## Pages 8–9 — Design Iteration Log

| Ver | Date | Change | Why | Impact |
|-----|------|--------|-----|--------|
| v0.1 | Aug 2025 | ~~CNN/video~~ → Pose-based | Video too slow for real-time | Enabled real-time processing |
| v0.2 | Sep 2025 | 27 → 75 keypoints | Standard landmarks miss finger detail | Better handshape discrimination |
| v0.3 | Sep 2025 | 16x → 50x augmentation | Insufficient training diversity | 342 → 17,100 samples |
| v0.4 | Oct 2025 | ~~Confidence-only~~ → LLM semantic coherence | Per-sign confidence ignores context | Quality Score 39.38 → 74.56 |
| v0.5 | Oct 2025 | Added CTQI framework | Single metrics miss communication quality | Multi-dimensional evaluation |
| v1.0 | Dec 2025 | Full pipeline integration | 5-step end-to-end pipeline | Video → English text working |
| v1.1 | Dec 2025 | Motion-based segmentation | Fixed-window misses sign boundaries | 85%+ boundary accuracy |
| v1.2 | Jan 2026 | 75 → 83 keypoints + model optimization | Final keypoint set + tuning | 80.97% Top-1, 91.62% Top-3 |
| v1.3 | Jan 2026 | Streaming demo + smart buffering | Real-time UX | 15-30 FPS, <500ms latency |
| v1.4 | Jan 2026 | LLM prompt iteration + error fixes | CCIR feedback + failure analysis | Improved edge case handling |
| v1.5 | Jan 2026 | CTQI v1 → v2 prerequisite chain | Better reflects translation quality | Multiplicative formula |
| v2.0 | Feb 2026 | Docker deployment to HuggingFace | Public access, reproducibility | Live demo online |

---

# Daily Engineering Entries

---

## Phase 1: Problem & Research (Aug 1 – Aug 31, 2025)

### Page 10 — Entry 1 — August 1, 2025

**Goal:** Define research problem and begin literature survey on ASL translation technology.

- Researched the Deaf community and existing solutions; first paper read: *The ASL Knowledge Graph* by Lee Kezar and Zed Seuyn
- Searched for esteemed, published papers related to ASL translation systems
- Found ~70M Deaf individuals worldwide, <20% digital content accurately captioned
- Identified three approach categories: video-based (slow), pose-based (fast, less accurate), hybrid
- Downloaded WLASL dataset docs — 2,000 sign classes, WLASL-100 has only 342 samples (~3.4/class)
- **Challenge:** Figuring out which sector of the problem to approach; narrowed from segmentation, pose-based recognition, or zero-shot accuracy to pose-based recognition; many dead-ends (zero-shot was a completely different skill set)
- **Reflection:** The many complexities in the overarching problem were overwhelming — so many possible problems and solutions

---

### Page 11 — Entry 2 — August 8, 2025

**Goal:** Study ASL linguistics — understand why translation is a grammar problem, not just recognition.

- Read WFD position paper; took notes by marking major points from abstract, then expanding while reading
- Watched short ASL videos to identify what needs to be analyzed for sign prediction

**Key ASL grammar features that break simple word-mapping:**
- Topic-Comment: `PIZZA, I LIKE` = "I like pizza"
- Time-first: `YESTERDAY STORE GO` = "I went to the store yesterday"
- Spatial referencing: `IX-a GIVE IX-b` = "She gave it to him"
- Non-manual markers: raised eyebrows = Y/N question
- Verb directionality: `GIVE-from-me-to-you` = "I give you"
- ASL is NOT "English on the hands" — it's a complete, independent language
- **Challenge:** Don't know ASL myself — relying on academic descriptions; hard to find large-scale parallel ASL-English corpora
- **Reflection:** Learning about ASL grammar made me realize I needed more than 1 metric and had to incorporate different measurement fields for an accurate evaluation system

---

### Page 12 — Entry 3 — August 15, 2025 | 3:00 PM

**Goal:** CCIR research training — problem statement workshop.

- One-on-one CCIR training on ML concepts and research paper writing; presented project idea
- Problem statement shifted from "build a sign language recognizer" to "bridge the communication gap between two grammatically distinct languages"
- **Challenge:** Kept jumping to technical solution before defining the problem
- **Reflection:** SignBridge isn't just a coding project — it's a research project needing hypotheses and experiments

---

### Page 13 — Entry 4 — August 22, 2025 | 3:00 PM

**Goal:** CCIR training — experimental design and methodology.

- Defined variables: Independent (keypoint count, augmentation factor, LLM vs no-LLM), Dependent (Top-1/Top-3 accuracy, CTQI, Quality Score), Controls (baseline OpenHands 27pts, confidence-only selection)
- Mentor suggested tracking Top-3 accuracy — if correct sign in Top-3, LLM could recover it
- CTQI idea came from mentor asking "how will you measure if the translation is good?" — couldn't answer with just accuracy
- **Reflection:** Shifted from developer thinking ("does it work?") to researcher thinking ("can I prove it works better?")

---

### Page 14 — Entry 5 — August 29, 2025 | 2:00 PM

**Goal:** Research synthesis — consolidate findings into project plan.

- Reviewed all August notes/papers; finalized RQ1–RQ3. Key benchmarks: I3D 65.89% (slow), Multi-stream CNN SOTA 81.38% (slow), OpenHands 71.57% (real-time), SignBridge target >80% (real-time)
- Pattern: existing approaches either achieve high accuracy but are too slow, or are fast but sacrifice accuracy
- No existing system addresses ASL-to-English grammar transformation — everyone stops at word-level recognition
- **Reflection:** August was about understanding the problem deeply before writing any code — this is a linguistics/communication problem that happens to use ML

---

## Phase 2: Design, Experiment & Prototype (Sep 1 – Nov 7, 2025)

### Page 15 — Entry 6 — September 5, 2025 | 4:00 PM

**Goal:** Design and implement OpenHands-HD keypoint expansion (83 points).

- Original OpenHands used 27 keypoints; designed OpenHands-HD: 8 face + 33 body + 42 hand (21L + 21R) = 83 keypoints → 279-dim features (3.4x more than original 81-dim)
- Chose based on ASL needs: hands critical, body provides context, face captures eyebrow position
- Still real-time (~30 FPS) despite 3x more keypoints; finger curl patterns now visible
- **Challenge:** MediaPipe returns None for hands out of frame — added handling; some low-res videos caused detection failure

---

### Page 16 — Entry 7 — September 15, 2025 | 3:30 PM

**Goal:** Build data augmentation pipeline (50x expansion).

- Implemented `generate_augmented_dataset.py` — rotation (±15°) and shear (±0.2 radians). Result: 342 → 17,100 samples; ~45 min generation; 30% held for validation/test
- **Challenge:** Initially tried 100x but model overfit; constrained rotation angles to prevent unrealistic poses

---

### Page 17 — Entry 8 — October 3, 2025 | 5:00 PM

**Goal:** Build initial 20-class and 50-class prototype models.

- Most frustrating period — ran 15+ failed experiments (CNN-LSTM at 1.5–10%, various balancing at 4–8%) before switching to OpenHands transformer
- **Data Leakage Bug (biggest mistake):** Augmented versions of same video in both train/val splits. Was getting 20% thinking it was real; fresh test data → 5%. Fixed split → 42.47% (8.5x jump)
- Final prototypes: 20-class 75pt augmented: 42.47% Top-1 / 75.29% Top-3; 50-class: 47.27% Top-1 / 67.25% Top-3
- **Reflection:** Top-3 >> Top-1 gap is the key finding — exactly what an LLM can exploit. All "progress" before fixing leakage was fake

---

### Page 18 — Entry 9 — October 15, 2025 | 4:30 PM

**Goal:** Design and implement LLM integration for sentence construction.

- Chose Gemini 2.0 Flash (fastest, free); built wrapper presenting Top-3 candidates per position for LLM to select coherent signs
- Early prompt failures: hallucinated extra signs, too verbose, picked high-confidence nonsense, wrong word order
- First success: MAN, DRINK, WATER → "The man wants to drink water" — LLM chose lower-confidence signs when they made more sense. Latency ~300-500ms
- **Reflection:** Prompt engineering was harder than expected — small wording changes have big effects

---

### Page 19 — Entry 10 — October 25, 2025 | 5:00 PM

**Goal:** Design the CTQI evaluation framework.

- Existing metrics (BLEU, BERTScore) didn't fully capture translation quality; designed CTQI as composite
- CTQI v2 formula: `(GA/100) × (CF1/100) × (0.5 + 0.5 × P/100) × 100` — multiplicative prerequisite chain forces all dimensions strong
- "time son go bed" scored ~55 vs "It is time for my son to go to bed" scored ~78 — captured exactly the quality difference
- **Challenge:** Switched from averaging to multiplicative because averaging let high accuracy mask bad grammar

---

### Page 20 — Entry 11 — November 5, 2025 | 6:00 PM

**Goal:** Prototype refinement and pre-paper baselines (50-class, 34 sentences). CF1 ~70→83%, Quality Score ~35→68, CTQI ~50→72. End-to-end working: video → English text. Decided to document current state and iterate after CCIR.

---

## Phase 3: Research Paper & CCIR Winter Symposium (Nov 8 – Dec 6, 2025)

### Page 21 — Entry 12 — November 8, 2025 | 4:00 PM

**Goal:** Begin CCIR paper. Structure: Abstract → Introduction → Related Work → Methodology → Results → Discussion → Conclusion → References.

---

### Page 22 — Entry 13 — November 24, 2025 | 11:00 PM

- Submitted paper to CCIR — ~8 pages, 3 figures, 4 tables.

---

### Page 23 — Entry 14 — November 28, 2025 | 3:00 PM

- Selected to present at CCIR Winter Symposium on December 6.

---

### Page 24 — Entry 15 — December 1–5, 2025 | 4:00 PM

**Goal:** Prepare CCIR presentation — 10 slides: hook → problem → architecture → results → live demo → future work. Timed for <10 minutes.

---

### Page 25 — Entry 16 — December 6, 2025 | 10:00 AM

**Goal:** Present at CCIR Winter Symposium. ~10 min talk + 5 min Q&A, ~20-30 person audience. Live demo was most popular; CTQI noted as novel contribution.

- **Lessons for CSRSEF:** Emphasize live demo, simplify CTQI explanation, prepare for Deaf community questions

---

## Phase 4: Build & Integrate (Dec 7 – Dec 24, 2025)

### Page 26 — Entry 17 — December 7, 2025 | 2:00 PM

**Goal:** Full pipeline integration. 5-step flow: Video → MediaPipe 83-pt extraction → Temporal segmentation → OpenHands-HD Top-K classification → Gemini LLM → English sentence. Latency <2s.

---

### Page 27 — Entry 18 — December 12, 2025 | 4:00 PM

**Goal:** Temporal segmentation — fixed-window and motion-based (hand velocity) methods. Boundary detection >85%, latency <200ms. Challenge: signs blend without clear pause.

---

### Page 28 — Entry 19 — December 18, 2025 | 3:00 PM

**Goal:** Pipeline testing — various inputs/lighting. Fixed race condition between pose extraction and model inference. Testing a multi-component pipeline is way harder than testing individual pieces.

*(Dec 25–30: Winter vacation)*

---

## Phase 5: Iterate, Optimize & Evaluate (Dec 31, 2025 – Jan 31, 2026)

### Page 29 — Entry 20 — December 31, 2025 | 1:00 PM

**Goal:** Real-time webcam demo — standard inference (file + webcam) and streaming inference (Gemini streaming API, smart buffering, 5 trigger strategies, 15-30 FPS, <500ms latency).

---

### Page 30 — Entry 21 — January 5, 2026 | 2:00 PM

**Goal:** Train optimized 100-class model. 10 training runs over ~2 weeks, progressively adding optimizations: SGD baseline 48.65% → AdamW 58.1% → Cosine LR 63.4% → batch 32 67.2% → warmup 71.8% → label smoothing 75.4% → gradient clipping 78.3% → final **80.97% Top-1, 91.62% Top-3**.

- Gloss list bug: manifest had 111 classes instead of 100; fixing jumped accuracy ~72→78%
- Final config: 3-layer transformer, 128 hidden, 8 heads, 279-dim, 1500 epochs, batch 32, dropout 0.3, label smoothing 0.1
- **Reflection:** 80.97% closes gap with video SOTA (81.38%) while being dramatically faster

---

### Page 31 — Entry 22 — January 10, 2026 | 3:00 PM

**Goal:** Full CTQI evaluation (n=34 sentences). CF1: 74.64→87.62 (p<0.001, d=0.848), Quality Score: 39.38→74.56 (p<0.001, d=1.149), CTQI: 55.56→78.16 (p<0.001, d=1.098), BLEU: 20.62→56.53, BERTScore: 91.64→96.30. 88.2% improved. ALL metrics p<0.001, Cohen's d >0.8; all hypotheses confirmed.

---

### Page 32 — Entry 23 — January 14, 2026 | 2:00 PM

**Goal:** Streaming inference — Gemini streaming API with conversation manager (5 trigger strategies). Latency <500ms.

---

### Page 33 — Entry 24 — January 18, 2026 | 4:00 PM

**Goal:** Model optimization. Overfitting was severe (Train 97%, Val 60%). Dropout 0.3 was sweet spot (Train 85%, Val 75%, 10% gap). Label smoothing 0.1 (+2-3%), Cosine LR, gradient clipping (prevented NaN crashes). 30+ hours across 2 weeks.

- **Reflection:** With limited data (342 base samples), regularization is MORE important than model capacity

---

### Page 34 — Entry 25 — January 21, 2026 | 3:00 PM

**Goal:** LLM prompt iteration — 6 versions over ~3 months. Biggest breakthrough: two-stage pipeline (Stage 1: Gloss Selection for coherence → Stage 2: Sentence Construction for grammar). Reduced hallucination ~20→8%. Key rule: "A sensible sentence beats a high-confidence nonsensical one." Final prompt: 11.6 KB, 8 examples.

---

### Page 35 — Entry 26 — January 24, 2026 | 2:00 PM

**Goal:** Incorporate CCIR feedback — improved error handling, simplified CTQI explanation, added Deaf community testing to future work.

---

### Page 36 — Entry 27 — January 28, 2026 | 5:00 PM

**Goal:** Final metrics — ablation showed LLM accounts for +22.60 CTQI (largest single contribution). Without LLM, accuracy same but communication quality tanks. All metrics p<0.001, Cohen's d >0.8.

---

### Page 36.5 — Entry 27.5 — January 30, 2026 | 4:00 PM

**Goal:** Redesign CTQI v1→v2. v1 (`0.4×GA + 0.4×Quality + 0.2×PTR`) had problems: PTR redundant with GA (r=0.855), arithmetic mean let 100% accuracy + terrible grammar score 60.0. v2 (`(GA/100) × (CF1/100) × (0.5 + 0.5×P/100) × 100`): multiplicative chain, all dimensions must be strong. v2 correctly penalized cases v1 masked.

---

## Phase 6: Deploy & Release (Feb 1 – Feb 13, 2026)

### Page 37 — Entry 28 — February 1, 2026 | 2:00 PM

**Goal:** Docker → HuggingFace Spaces. URL: https://huggingface.co/spaces/SuperiorHaxs/SignBridge. Publicly accessible, slow on cold start but smooth once warmed.

---

### Page 38 — Entry 29 — February 5, 2026 | 3:00 PM

**Goal:** Deployed app testing — works on Chrome/Firefox; Safari had webcam issues; mobile cramped but functional.

---

### Page 38.5 — Entry 29.5 — February 7, 2026 | 3:00 PM

**Goal:** Human survey to validate CTQI against real judgment. Google Forms, 53 sentences, 5 raters (family/friends/classmates), 0-5 scale. Raters didn't know how CTQI works.

---

### Page 38.7 — Entry 29.7 — February 9, 2026 | 4:00 PM

**Goal:** Analyze survey results. **Shock: CTQI v2 (r=0.7750) correlated WORSE with humans than v1 (r=0.9381).** Core problem: v2's plausibility gave full credit to grammatically correct but semantically WRONG translations (e.g., COMPUTER PLAY BASKETBALL → "My son plays basketball" — fluent but wrong signs).

- **Reflection:** Theory and human perception don't always align. Always validate against ground truth.

---

### Page 38.8 — Entry 29.8 — February 10, 2026 | 2:00 PM

**Goal:** Design CTQI v3. Insight: fluency should only count if signs are correct. One-line change: `P/100` → `P/100 × GA/100`. v3 formula: `(GA/100) × (CF1/100) × (0.5 + 0.5 × P/100 × GA/100) × 100`. **CTQI v3 achieves highest human correlation of ALL metrics (r=0.9427)**, beating v1 (+0.005), BLEU (+0.184), BERTScore (+0.135).

- **Reflection:** v2's theoretical elegance → empirical failure → v3's data-driven fix — most important thing I learned

---

## Phase 7: Science Fair Prep (Feb 14 – Mar 6, 2026)

### Page 39 — Entry 30 — February 14, 2026 | 4:00 PM

**Goal:** Begin CSRSEF materials — poster design in Canva, notebook printing into physical binder.

---

### Page 40 — Entry 31 — February 2026 | 6:00 PM

**Goal:** Demo rehearsal — 5 prepared sequences (MAN→DRINK→WATER, TIME→SON→GO→BED, WOMAN→WANT→EAT→FOOD, BOY→PLAY→BALL→HAPPY, GIRL→READ→BOOK→SLOW). Made offline fallback for internet dependency.

---

### Page 41 — Entry 32 — March 5–6, 2026 | 7:00 PM

**Goal:** Final preparations. Checklist complete: poster printed, hardcopy notebook, ISEF forms, laptop/webcam tested, demo working, USB backup.

---

## Page 42 — Code Repository Reference

**GitHub:** https://github.com/SuperiorHaxs/SignBridge
**Live Demo:** https://huggingface.co/spaces/SuperiorHaxs/SignBridge
**Commits:** 125 | **Languages:** Python 86.6%, JavaScript 7.0%, CSS 3.7%, HTML 2.6%

**Key files:** `models/training-scripts/train_asl.py` (training), `models/openhands-modernized/src/` (inference), `dataset-utilities/augmentation/generate_augmented_dataset.py` (50x augmentation), `dataset-utilities/landmarks-extraction/` (83-pt extraction), `applications/predict_sentence.py` (full pipeline), `applications/predict_sentence_with_gemini_streaming.py` (streaming), `applications/gemini_conversation_manager.py` (LLM buffering), `applications/motion_based_segmenter.py` (segmentation), `applications/show-and-tell/` (web demo), `project-utilities/evaluation_metrics/` (CTQI/BLEU/BERTScore), `project-utilities/llm_interface/` (multi-provider LLM), `Dockerfile` (deployment)

**Figures:** Fig 1 — Architecture (p.6), Fig 2 — Accuracy comparison (p.30), Fig 3 — Baseline vs LLM (p.31), Fig 4 — CTQI distribution (p.31), Fig 5–9 — Pose skeletons & demo GIFs (p.40), Fig 10–12 — Metric-human correlation & scatter plots (p.38.7–38.8)

---

*Last updated: February 2026*
*CSRSEF: March 7, 2026*
*Category: TECA-HIE*
*Pages 43–86 reserved for future work*
