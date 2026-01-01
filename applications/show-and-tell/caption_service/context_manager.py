"""
Context Manager for Caption Service

Manages the context window for real-time ASL captioning:
- Running caption (current accumulated text)
- Sentence history (previous complete sentences)
- Gloss history (all detected glosses in order)
- Reconstruction logic
"""

import threading
from typing import List, Dict, Optional, Any
from .config import (
    CONTEXT_MODE,
    MAX_CONTEXT_SENTENCES,
    RECONSTRUCTION_WINDOW_SIZE,
    CLARIFYING_WORDS,
)


class ContextManager:
    """
    Manages context for real-time caption generation.

    Thread-safe implementation for use with background processing.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Clear all context state."""
        with self._lock:
            self._running_caption = ""
            self._sentence_history: List[str] = []
            self._gloss_history: List[Dict[str, Any]] = []
            self._processed_count = 0
            self._last_translation_time = 0.0

    # =========================================================================
    # PROPERTIES (thread-safe getters)
    # =========================================================================

    @property
    def running_caption(self) -> str:
        """Get current running caption."""
        with self._lock:
            return self._running_caption

    @running_caption.setter
    def running_caption(self, value: str):
        """Set current running caption."""
        with self._lock:
            self._running_caption = value

    @property
    def sentence_history(self) -> List[str]:
        """Get copy of sentence history."""
        with self._lock:
            return self._sentence_history.copy()

    @property
    def gloss_history(self) -> List[Dict[str, Any]]:
        """Get copy of gloss history."""
        with self._lock:
            return self._gloss_history.copy()

    @property
    def processed_count(self) -> int:
        """Get number of processed glosses."""
        with self._lock:
            return self._processed_count

    @processed_count.setter
    def processed_count(self, value: int):
        """Set number of processed glosses."""
        with self._lock:
            self._processed_count = value

    @property
    def last_translation_time(self) -> float:
        """Get timestamp of last translation."""
        with self._lock:
            return self._last_translation_time

    @last_translation_time.setter
    def last_translation_time(self, value: float):
        """Set timestamp of last translation."""
        with self._lock:
            self._last_translation_time = value

    # =========================================================================
    # GLOSS MANAGEMENT
    # =========================================================================

    def add_gloss(self, gloss_data: Dict[str, Any]):
        """
        Add a single gloss to history.

        Args:
            gloss_data: Dict with 'gloss', 'confidence', 'top_k' keys
        """
        with self._lock:
            self._gloss_history.append(gloss_data)

    def add_glosses(self, glosses: List[Dict[str, Any]]):
        """
        Add multiple glosses to history.

        Args:
            glosses: List of gloss data dicts
        """
        with self._lock:
            self._gloss_history.extend(glosses)

    def get_new_glosses(self) -> List[Dict[str, Any]]:
        """
        Get glosses that haven't been processed yet.

        Returns:
            List of new gloss data dicts
        """
        with self._lock:
            return self._gloss_history[self._processed_count:]

    def get_gloss_count(self) -> int:
        """Get total number of glosses in history."""
        with self._lock:
            return len(self._gloss_history)

    # =========================================================================
    # SENTENCE MANAGEMENT
    # =========================================================================

    def add_sentence(self, sentence: str):
        """
        Add a completed sentence to history.

        Args:
            sentence: The completed sentence
        """
        with self._lock:
            self._sentence_history.append(sentence)

    def get_sentence_context(self) -> List[str]:
        """
        Get sentence context based on configured mode.

        Returns:
            List of previous sentences for context
        """
        with self._lock:
            if not self._sentence_history:
                return []

            if CONTEXT_MODE == "full_history":
                return self._sentence_history.copy()
            elif CONTEXT_MODE == "rolling_window":
                if MAX_CONTEXT_SENTENCES > 0:
                    return self._sentence_history[-MAX_CONTEXT_SENTENCES:]
                else:
                    return []
            else:
                return []

    # =========================================================================
    # RECONSTRUCTION LOGIC
    # =========================================================================

    def should_reconstruct(self, new_glosses: List[Dict[str, Any]]) -> bool:
        """
        Determine if caption should be reconstructed from scratch.

        Used in "smart" reconstruction mode to detect when new glosses
        contain clarifying words that might change interpretation.

        Args:
            new_glosses: The new glosses being added

        Returns:
            True if reconstruction is recommended
        """
        for gloss_data in new_glosses:
            # Check main gloss
            gloss = gloss_data.get('gloss', '').lower()
            if gloss in CLARIFYING_WORDS:
                return True

            # Check top-k predictions
            top_k = gloss_data.get('top_k', [])
            for pred in top_k[:3]:
                pred_gloss = pred.get('gloss', '').lower()
                if pred_gloss in CLARIFYING_WORDS:
                    return True

        return False

    def get_reconstruction_glosses(self) -> List[Dict[str, Any]]:
        """
        Get glosses to use for reconstruction (sliding window).

        Returns:
            List of recent glosses for reconstruction
        """
        with self._lock:
            if RECONSTRUCTION_WINDOW_SIZE > 0:
                return self._gloss_history[-RECONSTRUCTION_WINDOW_SIZE:]
            else:
                return self._gloss_history.copy()

    # =========================================================================
    # CONTEXT FOR PROMPT BUILDING
    # =========================================================================

    def get_context_for_prompt(
        self,
        new_glosses: Optional[List[Dict[str, Any]]] = None,
        is_reconstruction: bool = False
    ) -> Dict[str, Any]:
        """
        Get context data for building LLM prompt.

        Args:
            new_glosses: New glosses to process (None = get from history)
            is_reconstruction: Whether this is a full reconstruction

        Returns:
            Dict with context data for prompt building:
            {
                'glosses': List of glosses to process,
                'running_caption': Current caption text,
                'sentence_history': Previous sentences,
                'is_reconstruction': Whether to reconstruct
            }
        """
        with self._lock:
            if is_reconstruction:
                # Use sliding window of all glosses
                glosses = self._gloss_history[-RECONSTRUCTION_WINDOW_SIZE:] \
                    if RECONSTRUCTION_WINDOW_SIZE > 0 else self._gloss_history.copy()
            else:
                # Use only new glosses
                glosses = new_glosses if new_glosses else self._gloss_history[self._processed_count:]

            return {
                'glosses': glosses,
                'running_caption': self._running_caption,
                'sentence_history': self._sentence_history.copy(),
                'is_reconstruction': is_reconstruction,
            }

    # =========================================================================
    # STATE SUMMARY
    # =========================================================================

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current context state.

        Returns:
            Dict with state information for debugging/display
        """
        with self._lock:
            return {
                'running_caption': self._running_caption,
                'sentence_count': len(self._sentence_history),
                'gloss_count': len(self._gloss_history),
                'processed_count': self._processed_count,
                'pending_count': len(self._gloss_history) - self._processed_count,
            }
