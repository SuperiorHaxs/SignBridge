# ASL Recognition Project - Weekly Work Log

**Project Duration:** September 1, 2024 - January 7, 2025 (~18 weeks)
**Total Hours:** ~75 hours

---

## Week 1: September 1-7, 2024
**Hours: 4**

**Tasks Completed:**
- Project planning and scope definition
- Literature review: WLASL dataset papers, OpenHands architecture, I3D baselines
- Identified key benchmarks: 65.89% (I3D), 81.38% (Multi-stream CNN SOTA), 30.6% (OpenHands pose-based)
- Initial project structure planning

**Key Decisions:**
- Focus on pose-based approach for real-time performance
- Target WLASL dataset for standardized benchmarking

---

## Week 2: September 8-14, 2024
**Hours: 4**

**Tasks Completed:**
- Environment setup: Python 3.11.9, PyTorch, MediaPipe
- Downloaded WLASL dataset (videos and annotations)
- Created initial project directory structure
- Set up git repository

**Deliverables:**
- Working development environment
- Dataset downloaded and organized

---

## Week 3: September 15-21, 2024
**Hours: 5**

**Tasks Completed:**
- Pose extraction pipeline using MediaPipe
- Created video-to-pose conversion scripts
- Built initial data loading utilities
- Processed videos to 27-point pose sequences

**Challenges:**
- MediaPipe hand tracking inconsistency
- Variable video lengths requiring padding/truncation

---

## Week 4: September 22-28, 2024
**Hours: 4**

**Tasks Completed:**
- Implemented CNN-LSTM baseline architecture
- Created 20-class subset for initial experiments
- First training runs (Exp 1-2)
- Results: ~1.5% accuracy (failed - barely above random)

**Learnings:**
- 27 keypoints insufficient for sign discrimination
- Need more sophisticated architecture or features

---

## Week 5: September 29 - October 5, 2024
**Hours: 4**

**Tasks Completed:**
- Debugged CNN-LSTM training issues
- Exp 3: Reduced architecture, achieved 10.91% on 20 classes
- Exp 4a-4d: 50-class experiments with class balancing
- Best result: 8.94% with class balance (Exp 4d)

**Key Insight:**
- Class imbalance significantly affects training
- CNN-LSTM architecture hitting ceiling

---

## Week 6: October 6-12, 2024
**Hours: 5**

**Tasks Completed:**
- Migrated to OpenHands transformer architecture
- Updated pose extraction to 75 keypoints (body + full hands)
- Created centralized config system (config/paths.py, settings.json)
- Initial OpenHands training: 20% accuracy on 20 classes

**Deliverables:**
- OpenHands model integration
- Centralized configuration system

---

## Week 7: October 13-19, 2024
**Hours: 4**

**Tasks Completed:**
- Developed augmentation pipeline (26 variants)
- Geometric transforms: rotation, shear, scale
- Temporal transforms: speed variation, frame dropping
- Spatial noise and translation augmentations

**Result:**
- Augmentation ready for integration with training

---

## Week 8: October 20-26, 2024
**Hours: 5**

**Tasks Completed:**
- Integrated augmentation with training pipeline
- 20-class with augmentation: 42.47% val accuracy (8.5x improvement!)
- Fixed data leakage issue in train/val splits
- 50-class training: 47.27% val accuracy

**Major Milestone:**
- First significant results beating baselines
- Data leakage fix critical for valid evaluation

---

## Week 9: October 27 - November 2, 2024
**Hours: 4**

**Tasks Completed:**
- Major documentation overhaul
- Research-oriented README with challenges/solutions
- Training results comparison table
- Synchronized terminology across codebase

**Deliverables:**
- Comprehensive README.md
- training_results_comp.md

---

## Week 10: November 3-9, 2024
**Hours: 3**

**Tasks Completed:**
- BLEU score calculator implementation
- Synthetic sentence generator for evaluation
- Auto-concatenation of glosses
- Case-insensitive comparison

**Deliverables:**
- BLEU evaluation framework
- Synthetic evaluation scripts

---

## Week 11: November 10-16, 2024
**Hours: 5**

**Tasks Completed:**
- LLM integration with Gemini API
- Smart buffering system (5 trigger strategies)
- Context-aware sentence construction
- Top-K prediction support for disambiguation

**Deliverables:**
- LLM-powered sentence construction module
- Streaming API integration

---

## Week 12: November 17-23, 2024
**Hours: 4**

**Tasks Completed:**
- CCIR research paper draft (v1)
- Literature comparison section
- Methodology documentation
- Results analysis

**Deliverables:**
- Research paper first draft

---

## Week 13: November 24-30, 2024
**Hours: 3**

**Tasks Completed:**
- Updated LLM prompts for research paper
- Evaluation script improvements
- Research paper revisions

---

## Week 14: December 1-7, 2024
**Hours: 3**

**Tasks Completed:**
- Planning for demo application
- Architecture design for real-time system
- Demo questions preparation

---

## Week 15: December 8-14, 2024
**Hours: 6**

**Tasks Completed:**
- Show & Tell demo application (Flask web app)
- Real-time ASL detection microservices
- Closed-captions mode implementation
- Model inference with auto-config loading
- Training enhancements: label smoothing, LR warmup, grad clipping

**Deliverables:**
- Show & Tell web application
- Closed-captions real-time translation
- Enhanced training pipeline

---

## Week 16: December 29, 2024 - January 1, 2025
**Hours: 7**

**Tasks Completed:**
- Clean-slate balanced augmentation pipeline
- Manifest-based dataset loading
- Two-phase augmentation approach
- Added 3D coordinates (z-axis) support
- Finger features (30 additional features)
- 83-point keypoint configuration
- 100-class model: 48.65% val accuracy

**Major Updates:**
- 3D pose coordinates for depth information
- Finger feature extraction for fine-grained discrimination

---

## Week 17: January 2-4, 2025
**Hours: 6**

**Tasks Completed:**
- Incremental training pipeline for class expansion
- Smart 3-stage gloss selector
- Train sample capping algorithm
- 43-class model training: 80.97% top-1, 91.62% top-3
- Production model saved

**Major Milestone:**
- 43-class model achieves 80.97% - best result to date
- Smart gloss selection for vocabulary expansion

---

## Week 18: January 5-7, 2025
**Hours: 4**

**Tasks Completed:**
- 100-class gloss list refinement (32 baseline + 68 new)
- Fixed gloss_list passthrough bug in splitting pipeline
- Manifest filtering for custom class configurations
- Re-ran preprocessing with corrected 100 classes
- Per-class accuracy analysis script

**Bug Fixes:**
- Gloss list not being passed to stratified splitting
- Manifest had 111 classes instead of 100

**Current Status:**
- 100-class training ready to resume with correct dataset
- All preprocessing scripts fixed for custom class lists

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Weeks** | 18 |
| **Total Hours** | ~75 |
| **Models Trained** | 15+ experiments |
| **Best 20-class** | 42.47% top-1, 75.29% top-3 |
| **Best 50-class** | 47.27% top-1, 50.91% top-3 |
| **Best 43-class** | 80.97% top-1, 91.62% top-3 |
| **Best 100-class** | 48.65% top-1 (in progress) |
| **Improvement over baseline** | 8.5x - 35.2x |

### Key Deliverables
1. OpenHands-based transformer model with 83-point keypoints
2. 26-variant augmentation pipeline
3. LLM-powered sentence construction
4. Show & Tell demo application
5. Closed-captions real-time translation
6. Smart gloss selection for vocabulary expansion
7. Comprehensive research documentation
