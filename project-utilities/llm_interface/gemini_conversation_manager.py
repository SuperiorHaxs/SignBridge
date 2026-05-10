# -*- coding: utf-8 -*-
"""
gemini_conversation_manager.py
Real-time Gemini conversation manager for ASL video conferencing

Handles:
- Streaming Gemini API responses
- Smart buffering and triggering
- Latency optimization
- Thread-safe communication with main video loop
"""

import queue
import threading
import time
import re
from collections import deque
from typing import Optional, Tuple, Dict, List


class GeminiConversationManager:
    """
    Manages real-time Gemini conversation for ASL video calls.

    Features:
    - Streaming responses for low latency
    - Smart triggering (pause detection, buffer size, question words)
    - Local fallback for common phrases
    - Latency tracking and optimization
    """

    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-flash'):
        """
        Initialize Gemini conversation manager.

        Args:
            api_key: Gemini API key
            model_name: Model to use (gemini-pro is standard, use gemini-1.5-pro-latest for newer)
        """
        # Import Gemini
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=api_key)

            # Use fast model with optimized config
            generation_config = {
                'max_output_tokens': 50,  # Short responses for video calls
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40
            }

            self.model = genai.GenerativeModel(
                model_name,
                generation_config=generation_config
            )
            self.chat = self.model.start_chat(history=[])

            print(f"[Gemini] Initialized with model: {model_name}")

        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            ) from e

        # Thread-safe queues
        self.sign_input_queue = queue.Queue()
        self.response_output_queue = queue.Queue(maxsize=1)

        # State management
        self.word_buffer = []
        self.last_sign_time = time.time()
        self.last_gemini_call = time.time()
        self.streaming_sentence = ""
        self.is_streaming = False

        # Latency tracking
        self.latency_stats = deque(maxlen=20)

        # Worker thread
        self.worker_thread = None
        self.stop_flag = threading.Event()

        # Configuration
        self.pause_threshold = 1.8  # seconds
        self.buffer_size_threshold = 3  # words
        self.max_timeout = 10.0  # seconds

    def start(self):
        """Start background worker thread"""
        self.worker_thread = threading.Thread(
            target=self._gemini_worker_thread,
            daemon=True
        )
        self.worker_thread.start()
        print("[Gemini] Worker thread started")

    def stop(self):
        """Stop background worker thread"""
        self.stop_flag.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print("[Gemini] Worker thread stopped")

    def add_signed_word(self, word: str, confidence: float):
        """
        Add a newly signed word to the buffer.

        Called by main prediction thread.

        Args:
            word: Predicted word/gloss
            confidence: Prediction confidence (0-1)
        """
        self.sign_input_queue.put({
            'word': word,
            'confidence': confidence,
            'timestamp': time.time()
        })
        self.last_sign_time = time.time()

    def get_display_text(self) -> Tuple[Optional[str], bool]:
        """
        Get current Gemini response for display.

        Called by main video loop (non-blocking).

        Returns:
            (response_text, is_complete) or (None, False) if no update
        """
        try:
            response = self.response_output_queue.get_nowait()
            return response['text'], response.get('complete', False)
        except queue.Empty:
            return None, False

    def get_avg_latency(self) -> float:
        """Get average Gemini API latency"""
        if self.latency_stats:
            return sum(self.latency_stats) / len(self.latency_stats)
        return 0.0

    def get_buffer_preview(self, max_words: int = 5) -> str:
        """Get preview of current word buffer"""
        words = [w['word'] for w in self.word_buffer[-max_words:]]
        return ' '.join(words)

    def _gemini_worker_thread(self):
        """Background worker thread - handles Gemini API calls"""
        while not self.stop_flag.is_set():
            try:
                # Collect words with timeout
                word_data = self.sign_input_queue.get(timeout=0.5)
                self.word_buffer.append(word_data)

                # Check if we should trigger Gemini
                should_call, reason = self._should_trigger_gemini()

                if should_call:
                    self._call_gemini_streaming(reason)

            except queue.Empty:
                # No new signs - check if buffer has stale data
                if self.word_buffer:
                    time_since_last = time.time() - self.last_sign_time
                    if time_since_last > self.pause_threshold:
                        self._call_gemini_streaming("pause_timeout")

    def _should_trigger_gemini(self) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should call Gemini API.

        Triggering strategies:
        1. Pause detection (1.5-2s silence with 2+ words)
        2. Buffer size (3-4 words accumulated)
        3. Question words (immediate response)
        4. Sentence enders
        5. Timeout (max 10s without response)

        Returns:
            (should_trigger, reason)
        """
        if not self.word_buffer:
            return False, None

        # Strategy 1: Pause detection
        time_since_last = time.time() - self.last_sign_time
        if time_since_last > self.pause_threshold and len(self.word_buffer) >= 2:
            return True, "pause"

        # Strategy 2: Buffer size threshold
        if len(self.word_buffer) >= self.buffer_size_threshold:
            return True, "buffer_full"

        # Strategy 3: Question words (immediate)
        last_word = self.word_buffer[-1]['word'].lower()
        if last_word in ['what', 'who', 'where', 'when', 'why', 'how']:
            if len(self.word_buffer) >= 2:
                return True, "question_word"

        # Strategy 4: Sentence enders
        if last_word in ['.', '?', '!', 'please', 'thanks', 'thank']:
            return True, "sentence_end"

        # Strategy 5: Timeout
        if time.time() - self.last_gemini_call > self.max_timeout:
            return True, "timeout"

        return False, None

    def _call_gemini_streaming(self, trigger_reason: str):
        """
        Call Gemini API with streaming response.

        Args:
            trigger_reason: Why Gemini was triggered
        """
        start_time = time.time()

        # Build signed text from buffer
        signed_text = ' '.join([w['word'] for w in self.word_buffer])
        avg_confidence = sum([w['confidence'] for w in self.word_buffer]) / len(self.word_buffer)

        print(f"[Gemini] Trigger: {trigger_reason} | Buffer: '{signed_text}'")

        # Try local fallback first (ultra-fast)
        local_response, _ = self._local_fallback(signed_text)
        if local_response:
            print(f"[Gemini] Local fallback: '{local_response}' (instant)")
            self.response_output_queue.put({
                'text': local_response,
                'complete': True,
                'trigger': 'local_fallback',
                'latency': 0.0
            })
            self.word_buffer.clear()
            self.last_gemini_call = time.time()
            return

        # Build context-aware prompt
        prompt = self._build_context_prompt(signed_text, avg_confidence, trigger_reason)

        try:
            self.is_streaming = True
            self.streaming_sentence = ""

            # Call Gemini (non-streaming, like predict_sentence.py does)
            # Streaming has StopIteration issues with this model/API version
            response = self.model.generate_content(prompt)

            # Get text from response
            if response and hasattr(response, 'text'):
                self.streaming_sentence = response.text.strip()
            else:
                self.streaming_sentence = ""

            # Mark complete
            total_latency = time.time() - start_time
            self.response_output_queue.put({
                'text': self.streaming_sentence,
                'complete': True,
                'trigger': trigger_reason,
                'latency': total_latency
            })

            # Track latency
            self.latency_stats.append(total_latency)

            print(f"[Gemini] Complete: {total_latency:.2f}s | '{self.streaming_sentence}'")

        except Exception as e:
            print(f"[Gemini Error] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.response_output_queue.put({
                'text': "[Gemini unavailable]",
                'complete': True,
                'error': str(e)
            })

        finally:
            self.is_streaming = False
            self.word_buffer.clear()
            self.last_gemini_call = time.time()

    def _build_context_prompt(self, signed_text: str, confidence: float, trigger: str) -> str:
        """
        Build context-aware prompt for Gemini.

        Args:
            signed_text: Words that were signed
            confidence: Average confidence
            trigger: Trigger reason

        Returns:
            Formatted prompt string
        """
        base_prompt = f"""You are assisting in a real-time ASL (American Sign Language) video call.
The user just signed: "{signed_text}"

Context:
- Confidence: {confidence:.2f}
- Trigger: {trigger}
- This is a live conversation, respond naturally and briefly (1-2 sentences max)

"""

        # Add specific instructions based on trigger
        if trigger == "question_word":
            base_prompt += "The user asked a question. Provide a helpful, concise answer.\n"
        elif trigger == "low_confidence":
            base_prompt += "Recognition confidence is low. Ask for clarification politely.\n"
        elif trigger in ["pause", "pause_timeout"]:
            base_prompt += "User paused. Acknowledge or continue naturally.\n"
        elif trigger == "sentence_end":
            base_prompt += "User completed a statement. Respond appropriately.\n"

        base_prompt += "\nRespond naturally as if in a video call:"

        return base_prompt

    def _local_fallback(self, signed_text: str) -> Tuple[Optional[str], float]:
        """
        Ultra-fast local response for common patterns.

        Returns:
            (response, latency) or (None, None) if no match
        """
        patterns = {
            r'^(hello|hi|hey)$': "Hello! Good to see you!",
            r'^how are you': "I'm good, thanks! How about you?",
            r'what.*time': f"It's {time.strftime('%I:%M %p')}",
            r'(thank.*you|thanks)$': "You're welcome!",
            r'^(bye|goodbye)$': "Goodbye! Take care!",
            r'^(yes|yeah)$': "Great!",
            r'^(no|nope)$': "Okay, understood.",
            r'^help': "How can I help you?",
            r'sorry': "No problem at all!",
        }

        signed_lower = signed_text.lower().strip()

        for pattern, response in patterns.items():
            if re.search(pattern, signed_lower):
                return response, 0.0

        return None, None


def wrap_text(text: str, max_chars: int = 50) -> List[str]:
    """
    Simple word wrapping for display.

    Args:
        text: Text to wrap
        max_chars: Maximum characters per line

    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += word + " "
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + " "

    if current_line:
        lines.append(current_line.strip())

    return lines


if __name__ == "__main__":
    # Test the manager
    import os

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        exit(1)

    manager = GeminiConversationManager(api_key)
    manager.start()

    # Simulate signing
    print("Simulating sign sequence...")
    manager.add_signed_word("hello", 0.95)
    time.sleep(0.5)
    manager.add_signed_word("how", 0.88)
    time.sleep(0.5)
    manager.add_signed_word("you", 0.92)

    # Wait for response
    print("\nWaiting for Gemini response...")
    time.sleep(3)

    text, complete = manager.get_display_text()
    if text:
        print(f"\nGemini response: '{text}'")
        print(f"Complete: {complete}")

    print(f"Avg latency: {manager.get_avg_latency():.2f}s")

    manager.stop()
