"""
Caption Service - Core Service Class

Real-time ASL closed-caption service that:
- Processes sign video blobs into gloss predictions
- Manages context window for coherent captions
- Calls LLM with context to construct sentences
- Supports file-based async communication

Designed to be standalone and embeddable in other applications.
"""

import os
import sys
import json
import time
import threading
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from .config import (
    GLOSS_FILE,
    SENTENCE_FILE,
    CAPTION_BUFFER_SIZE,
    FILE_CHECK_INTERVAL,
    MIN_TRANSLATION_DELAY,
    ENABLE_INCREMENTAL_MODE,
    ENABLE_DYNAMIC_RECONSTRUCTION,
    RECONSTRUCTION_MODE,
    LLM_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)
from .context_manager import ContextManager


class CaptionService:
    """
    Real-time ASL caption service.

    Processes signs, manages context, and generates English captions.
    Can be used standalone or integrated into Flask/other applications.
    """

    def __init__(
        self,
        session_dir: Path,
        model: Any = None,
        tokenizer: Any = None,
        llm_provider: Any = None,
        prompt_template: Optional[str] = None,
        on_caption_update: Optional[Callable[[str], None]] = None,
        on_gloss_detected: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize Caption Service.

        Args:
            session_dir: Directory for session files (glosses.txt, sentence.txt)
            model: OpenHands model for gloss prediction (optional, can be set later)
            tokenizer: Model tokenizer (optional)
            llm_provider: LLM provider instance (optional, will create default)
            prompt_template: Custom prompt template (optional)
            on_caption_update: Callback when caption updates (optional)
            on_gloss_detected: Callback when gloss detected (optional)
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.tokenizer = tokenizer
        self.llm_provider = llm_provider
        self.prompt_template = prompt_template

        # Callbacks
        self.on_caption_update = on_caption_update
        self.on_gloss_detected = on_gloss_detected

        # File paths
        self.gloss_file = self.session_dir / GLOSS_FILE
        self.sentence_file = self.session_dir / SENTENCE_FILE

        # Context manager
        self.context = ContextManager()

        # Current gloss status (used/dropped) for frontend display
        self.current_gloss_status = []

        # Threading
        self._file_watcher_running = False
        self._file_watcher_thread = None
        self._file_lock = threading.Lock()

        # Lazy load LLM provider
        self._llm_initialized = False

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Start the background file watcher thread."""
        if self._file_watcher_running:
            return

        self._file_watcher_running = True
        self._file_watcher_thread = threading.Thread(
            target=self._file_watcher_loop,
            daemon=True
        )
        self._file_watcher_thread.start()
        print(f"[CaptionService] Started file watcher for {self.session_dir}")

    def stop(self):
        """Stop the background file watcher thread."""
        self._file_watcher_running = False
        if self._file_watcher_thread:
            self._file_watcher_thread.join(timeout=2.0)
            self._file_watcher_thread = None

        # Flush any remaining unprocessed glosses
        self._flush_remaining_glosses()
        print("[CaptionService] Stopped file watcher")

    def _flush_remaining_glosses(self):
        """Process any remaining glosses that haven't reached buffer threshold."""
        try:
            all_glosses = self._read_glosses_from_file()
            current_count = len(all_glosses)
            processed_count = self.context.processed_count
            remaining = current_count - processed_count

            if remaining > 0:
                print(f"[CaptionService] Flushing {remaining} remaining glosses (total: {current_count})")

                self.context._gloss_history = all_glosses

                # Use ALL glosses for final caption (not limited by reconstruction window)
                # Pass is_reconstruction=False to avoid window limit, but pass all glosses
                sentence = self._construct_sentence(
                    new_glosses=all_glosses,  # Use ALL glosses, not just remaining
                    is_reconstruction=False
                )

                # Update context
                if ENABLE_INCREMENTAL_MODE:
                    self.context.running_caption = sentence

                self.context.add_sentence(sentence)
                self.context.processed_count = current_count

                # Write to file
                self._write_sentence_to_file(sentence)

                # Callback
                if self.on_caption_update:
                    self.on_caption_update(sentence)

                print(f"[CaptionService] Final caption: {sentence}")

        except Exception as e:
            print(f"[CaptionService] Error flushing remaining glosses: {e}")
            import traceback
            traceback.print_exc()

    def reset(self):
        """Clear all state and files."""
        self.context.reset()
        self.current_gloss_status = []  # Clear gloss status

        # Clear files
        try:
            if self.gloss_file.exists():
                self.gloss_file.unlink()
            if self.sentence_file.exists():
                self.sentence_file.unlink()
        except Exception as e:
            print(f"[CaptionService] Error clearing files: {e}")

        print("[CaptionService] Session reset")

    # =========================================================================
    # LLM INITIALIZATION
    # =========================================================================

    def _ensure_llm(self):
        """Lazy initialize LLM provider."""
        if self._llm_initialized:
            return

        llm_ok = True

        if self.llm_provider is None:
            try:
                # Import from project utilities
                llm_path = str(Path(__file__).parent.parent.parent.parent / "project-utilities" / "llm_interface")
                if llm_path not in sys.path:
                    sys.path.insert(0, llm_path)
                from llm_factory import create_llm_provider
                self.llm_provider = create_llm_provider(
                    provider=LLM_PROVIDER,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=LLM_TIMEOUT
                )
                print(f"[CaptionService] Initialized LLM provider: {LLM_PROVIDER}")
            except Exception as e:
                print(f"[CaptionService] Failed to initialize LLM: {e}")
                import traceback
                traceback.print_exc()
                llm_ok = False

        if self.prompt_template is None:
            try:
                # Use closed captions specific prompt (allows skipping nonsensical predictions)
                prompt_path = Path(__file__).parent.parent.parent.parent / "project-utilities" / "llm_interface" / "prompts" / "llm_prompt_closed_captions.txt"
                if prompt_path.exists():
                    self.prompt_template = prompt_path.read_text(encoding='utf-8')
                    print(f"[CaptionService] Loaded closed captions prompt template")
                else:
                    print(f"[CaptionService] Prompt file not found: {prompt_path}")
                    llm_ok = False
            except Exception as e:
                print(f"[CaptionService] Failed to load prompt template: {e}")
                llm_ok = False

        # Only mark as initialized if both provider and prompt are ready
        # This allows retry on next call if something failed
        self._llm_initialized = llm_ok

    # =========================================================================
    # SIGN PROCESSING
    # =========================================================================

    def process_sign(
        self,
        video_bytes: bytes,
        predict_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a sign video blob and return prediction.

        Args:
            video_bytes: Raw video data (WebM/MP4)
            predict_fn: Custom prediction function (uses self.model if None)

        Returns:
            Dict with:
            {
                'gloss': str,
                'confidence': float,
                'top_k': List[Dict],
                'success': bool
            }
        """
        if predict_fn is None and self.model is None:
            return {
                'success': False,
                'error': 'No model or predict_fn provided',
                'gloss': 'UNKNOWN',
                'confidence': 0.0,
                'top_k': []
            }

        try:
            # Save video to temp file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                f.write(video_bytes)
                temp_video_path = f.name

            # Process video to pose and predict
            if predict_fn:
                result = predict_fn(temp_video_path)
            else:
                result = self._default_predict(temp_video_path)

            # Clean up
            try:
                os.unlink(temp_video_path)
            except:
                pass

            # Add to gloss history
            gloss_data = {
                'gloss': result.get('gloss', 'UNKNOWN'),
                'confidence': result.get('confidence', 0.0),
                'top_k': result.get('top_k_predictions', result.get('top_k', []))[:3]
            }

            self.context.add_gloss(gloss_data)

            # Add to current_gloss_status as pending (will be updated when LLM processes)
            self.current_gloss_status.append({
                'gloss': gloss_data['gloss'],
                'confidence': gloss_data['confidence'],
                'used': None  # None = pending, True = used, False = dropped
            })

            # Write to file
            self._write_gloss_to_file(gloss_data)

            # Callback
            if self.on_gloss_detected:
                self.on_gloss_detected(gloss_data)

            return {
                'success': True,
                'gloss': gloss_data['gloss'],
                'confidence': gloss_data['confidence'],
                'top_k': gloss_data['top_k']
            }

        except Exception as e:
            print(f"[CaptionService] Error processing sign: {e}")
            return {
                'success': False,
                'error': str(e),
                'gloss': 'UNKNOWN',
                'confidence': 0.0,
                'top_k': []
            }

    def _default_predict(self, video_path: str) -> Dict[str, Any]:
        """
        Default prediction using OpenHands model.

        Override or provide predict_fn for custom prediction logic.
        """
        # This would need the full inference pipeline
        # For now, return placeholder
        return {
            'gloss': 'UNKNOWN',
            'confidence': 0.0,
            'top_k_predictions': []
        }

    # =========================================================================
    # FILE I/O
    # =========================================================================

    def _write_gloss_to_file(self, gloss_data: Dict[str, Any]):
        """Write gloss to file asynchronously (non-blocking)."""
        def write_async():
            try:
                with self._file_lock:
                    with open(self.gloss_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(gloss_data) + '\n')
            except Exception as e:
                print(f"[CaptionService] Error writing gloss: {e}")

        threading.Thread(target=write_async, daemon=True).start()

    def _read_glosses_from_file(self) -> list:
        """Read all glosses from file."""
        glosses = []
        try:
            if self.gloss_file.exists():
                with self._file_lock:
                    with open(self.gloss_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    glosses.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
        except Exception as e:
            print(f"[CaptionService] Error reading glosses: {e}")
        return glosses

    def _write_sentence_to_file(self, sentence: str):
        """Write sentence to file."""
        try:
            with self._file_lock:
                with open(self.sentence_file, 'w', encoding='utf-8') as f:
                    f.write(sentence)
        except Exception as e:
            print(f"[CaptionService] Error writing sentence: {e}")

    # =========================================================================
    # FILE WATCHER
    # =========================================================================

    def _file_watcher_loop(self):
        """Background loop that watches gloss file and triggers translation."""
        print(f"[CaptionService] File watcher loop started (buffer_size={CAPTION_BUFFER_SIZE}, check_interval={FILE_CHECK_INTERVAL}s)")
        print(f"[CaptionService] Watching gloss file: {self.gloss_file}")

        _last_log_count = 0

        while self._file_watcher_running:
            try:
                # Read all glosses from file
                all_glosses = self._read_glosses_from_file()
                current_count = len(all_glosses)
                processed_count = self.context.processed_count

                # Calculate new glosses
                new_gloss_count = current_count - processed_count

                # Log when new glosses appear
                if current_count != _last_log_count:
                    print(f"[CaptionService] File has {current_count} glosses, {processed_count} processed, {new_gloss_count} new (need {CAPTION_BUFFER_SIZE})")
                    _last_log_count = current_count

                if new_gloss_count >= CAPTION_BUFFER_SIZE:
                    # Check cooldown
                    time_since_last = time.time() - self.context.last_translation_time

                    if time_since_last < MIN_TRANSLATION_DELAY:
                        time.sleep(FILE_CHECK_INTERVAL)
                        continue

                    print(f"[CaptionService] Processing {new_gloss_count} new glosses")

                    # Get new glosses
                    new_glosses = all_glosses[processed_count:]

                    # Sync context with file
                    self.context._gloss_history = all_glosses

                    # Determine reconstruction mode
                    should_reconstruct = False
                    if ENABLE_INCREMENTAL_MODE and ENABLE_DYNAMIC_RECONSTRUCTION:
                        if self.context.running_caption:
                            if RECONSTRUCTION_MODE == "sliding_window":
                                should_reconstruct = True
                            elif RECONSTRUCTION_MODE == "smart":
                                should_reconstruct = self.context.should_reconstruct(new_glosses)

                    # Construct sentence
                    sentence = self._construct_sentence(
                        new_glosses=new_glosses,
                        is_reconstruction=should_reconstruct
                    )

                    # Update context
                    if ENABLE_INCREMENTAL_MODE:
                        self.context.running_caption = sentence

                    self.context.add_sentence(sentence)
                    self.context.processed_count = current_count
                    self.context.last_translation_time = time.time()

                    # Write to file
                    self._write_sentence_to_file(sentence)

                    # Callback
                    if self.on_caption_update:
                        self.on_caption_update(sentence)

                    print(f"[CaptionService] Caption: {sentence}")

            except Exception as e:
                print(f"[CaptionService] File watcher error: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(FILE_CHECK_INTERVAL)

        print("[CaptionService] File watcher loop ended")

    # =========================================================================
    # SENTENCE CONSTRUCTION
    # =========================================================================

    def _construct_sentence(
        self,
        new_glosses: list,
        is_reconstruction: bool = False
    ) -> str:
        """
        Construct English sentence from glosses using LLM.

        Args:
            new_glosses: New glosses to process
            is_reconstruction: Whether to reconstruct from all glosses

        Returns:
            Constructed English sentence
        """
        self._ensure_llm()

        if not self.llm_provider or not self.prompt_template:
            # Fallback: just join glosses
            gloss_words = [g.get('gloss', 'UNKNOWN') for g in new_glosses]
            # Mark all as used in fallback mode
            self.current_gloss_status = [
                {'gloss': g.get('gloss', 'UNKNOWN'), 'confidence': g.get('confidence', 0), 'used': True}
                for g in new_glosses
            ]
            return ' '.join(gloss_words)

        try:
            # Get context
            ctx = self.context.get_context_for_prompt(
                new_glosses=new_glosses,
                is_reconstruction=is_reconstruction
            )

            # Build prompt
            prompt = self._build_prompt_with_context(
                glosses=ctx['glosses'],
                running_caption=ctx['running_caption'] if ENABLE_INCREMENTAL_MODE else None,
                sentence_history=ctx['sentence_history'],
                is_reconstruction=is_reconstruction
            )

            # Call LLM
            response = self.llm_provider.generate(prompt)

            # Parse response (now returns dict with sentence and gloss_status)
            result = self._parse_llm_response(response, ctx['glosses'])

            # Update gloss status for glosses that were processed
            # Match by gloss name and update used status
            selection_set = set(s.lower() for s in result['selections'])
            for gloss_entry in self.current_gloss_status:
                if gloss_entry['used'] is None:  # Only update pending ones
                    gloss_entry['used'] = gloss_entry['gloss'].lower() in selection_set

            return result['sentence']

        except Exception as e:
            print(f"[CaptionService] LLM error: {e}")
            # Fallback
            gloss_words = [g.get('gloss', 'UNKNOWN') for g in new_glosses]
            return ' '.join(gloss_words)

    def _build_prompt_with_context(
        self,
        glosses: list,
        running_caption: Optional[str],
        sentence_history: list,
        is_reconstruction: bool
    ) -> str:
        """Build LLM prompt with context section."""
        # Format gloss details
        gloss_details = self._format_gloss_details(glosses)

        # Build context section
        context_section = ""

        if is_reconstruction and running_caption:
            context_section = f"""
Previous Caption (for reference - you may completely rewrite):
"{running_caption}"

RECONSTRUCTION MODE: Create the best possible caption from ALL glosses below.
You may completely change the previous interpretation if it makes more sense.
"""
        elif ENABLE_INCREMENTAL_MODE and running_caption:
            context_section = f"""
Current Running Caption (extend this):
"{running_caption}"

INCREMENTAL MODE: Add the new words to extend the caption naturally.
"""

        if sentence_history and not is_reconstruction:
            context_section += "\nPrevious Sentences (for context):\n"
            for i, sent in enumerate(sentence_history[-2:], 1):
                context_section += f"  {i}. \"{sent}\"\n"

        # Insert context into template
        if '{context_section}' in self.prompt_template:
            prompt = self.prompt_template.replace('{context_section}', context_section)
        else:
            # Prepend context if no placeholder
            prompt = context_section + "\n" + self.prompt_template

        # Insert gloss details
        prompt = prompt.replace('{gloss_details}', gloss_details)

        return prompt

    def _format_gloss_details(self, glosses: list) -> str:
        """Format glosses for LLM prompt."""
        details = []
        for i, g in enumerate(glosses, 1):
            top_k = g.get('top_k', [])
            if top_k:
                detail = f"Position {i}:\n"
                for j, pred in enumerate(top_k[:3], 1):
                    conf = pred.get('confidence', 0) * 100
                    detail += f"  Option {j}: '{pred.get('gloss', 'UNKNOWN')}' (confidence: {conf:.1f}%)\n"
                details.append(detail)
            else:
                conf = g.get('confidence', 0) * 100
                details.append(f"Position {i}: '{g.get('gloss', 'UNKNOWN')}' (confidence: {conf:.1f}%)\n")
        return "".join(details)

    def _parse_llm_response(self, response: str, glosses: list) -> Dict[str, Any]:
        """Parse LLM response to extract sentence and selections."""
        import re

        response_text = response.strip()

        # Handle markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        sentence = ""
        selections = []

        # Try to parse JSON
        try:
            result = json.loads(response_text)
            sentence = result.get('sentence', response_text)
            selections = result.get('selections', [])
        except json.JSONDecodeError:
            # Fallback: Try to find JSON object in response
            try:
                json_match = re.search(r'\{[^{}]*"sentence"\s*:\s*"([^"]+)"[^{}]*\}', response_text)
                if json_match:
                    sentence = json_match.group(1)
                # Try to extract selections
                sel_match = re.search(r'"selections"\s*:\s*\[([^\]]+)\]', response_text)
                if sel_match:
                    selections = [s.strip().strip('"\'') for s in sel_match.group(1).split(',')]
            except Exception:
                pass

        # If no sentence found, use raw text
        if not sentence:
            sentence = response_text
            print(f"[CaptionService] Warning: Could not parse LLM response: {response_text[:100]}")

        # Mark glosses as used or dropped
        gloss_status = []
        selection_set = set(s.lower() for s in selections)
        for g in glosses:
            gloss_word = g.get('gloss', 'UNKNOWN').lower()
            used = gloss_word in selection_set
            gloss_status.append({
                'gloss': g.get('gloss', 'UNKNOWN'),
                'confidence': g.get('confidence', 0),
                'used': used
            })

        return {
            'sentence': sentence,
            'selections': selections,
            'gloss_status': gloss_status
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_current_caption(self) -> str:
        """Get current caption text."""
        return self.context.running_caption

    def get_sentence_file_content(self) -> str:
        """Read current sentence from file."""
        try:
            if self.sentence_file.exists():
                return self.sentence_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"[CaptionService] Error reading sentence file: {e}")
        return ""

    def get_gloss_status(self) -> list:
        """Get current gloss status (used/dropped for each gloss)."""
        return self.current_gloss_status

    def get_state(self) -> Dict[str, Any]:
        """Get current service state."""
        return {
            'running_caption': self.context.running_caption,
            'gloss_count': self.context.get_gloss_count(),
            'processed_count': self.context.processed_count,
            'sentence_count': len(self.context.sentence_history),
            'is_running': self._file_watcher_running,
            'gloss_status': self.current_gloss_status,
        }
