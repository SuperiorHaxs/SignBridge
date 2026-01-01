"""
Caption Service Configuration

Configuration constants for real-time ASL closed-captions.
"""

# =============================================================================
# FILE PATHS (relative to session directory)
# =============================================================================
GLOSS_FILE = "detected_glosses.txt"
SENTENCE_FILE = "translated_sentence.txt"

# =============================================================================
# TIMING CONFIGURATION
# =============================================================================
# Minimum glosses before triggering LLM translation
CAPTION_BUFFER_SIZE = 3

# Interval between file checks (seconds)
FILE_CHECK_INTERVAL = 0.5

# Minimum delay between translations (prevents rapid-fire API calls)
MIN_TRANSLATION_DELAY = 2.0

# =============================================================================
# CONTEXT WINDOW CONFIGURATION
# =============================================================================
# Context modes: "full_history", "rolling_window"
CONTEXT_MODE = "full_history"

# Number of previous sentences to include in context (for rolling_window mode)
MAX_CONTEXT_SENTENCES = 2

# Enable incremental caption building (extend running caption)
ENABLE_INCREMENTAL_MODE = True

# Enable dynamic reconstruction (rewrite entire caption when needed)
ENABLE_DYNAMIC_RECONSTRUCTION = True

# Reconstruction modes: "sliding_window", "smart"
# - sliding_window: Always use last N glosses
# - smart: Only reconstruct when clarifying words detected
RECONSTRUCTION_MODE = "sliding_window"

# Number of glosses to use for reconstruction (sliding window)
RECONSTRUCTION_WINDOW_SIZE = 6

# =============================================================================
# MOTION DETECTION CONFIGURATION (browser-side, passed to frontend)
# =============================================================================
MOTION_CONFIG = {
    'cooldown_ms': 1000,           # No motion = end sign
    'min_sign_ms': 500,            # Minimum sign duration
    'max_sign_ms': 5000,           # Maximum before force-end
    'motion_threshold': 30,        # Pixel difference threshold
    'motion_area_threshold': 0.02, # 2% of frame must change
}

# =============================================================================
# CLARIFYING WORDS (trigger reconstruction in "smart" mode)
# =============================================================================
CLARIFYING_WORDS = {
    # Question words
    'who', 'what', 'which', 'that', 'this', 'these', 'those',
    # Family/person references
    'mother', 'father', 'man', 'woman', 'doctor', 'cousin',
    'brother', 'sister', 'son', 'daughter', 'friend',
    # Temporal markers
    'now', 'before', 'finish', 'year', 'yesterday', 'tomorrow',
    # WH-questions
    'where', 'when', 'why', 'how',
    # Negation
    'no', 'not', 'never', 'wrong',
    # Quantifiers
    'all', 'many', 'few', 'some',
}

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
LLM_PROVIDER = "googleaistudio"
LLM_MAX_TOKENS = 500
LLM_TIMEOUT = 30
