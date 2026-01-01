"""
Incremental Training Pipeline

Tools for iteratively building a high-accuracy ASL recognition model:
1. analyze_per_class_accuracy.py - Compute per-class validation accuracy
2. compute_gloss_embeddings.py - Extract model embeddings for candidate glosses
3. select_next_glosses.py - Rank and select next glosses to add
4. incremental_trainer.py - Orchestrate the incremental training process
"""
