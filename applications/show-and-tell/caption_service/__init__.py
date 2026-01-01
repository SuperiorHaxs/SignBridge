"""
Caption Service Module

Real-time ASL closed-caption service for show-and-tell and other applications.

Usage:
    from caption_service import CaptionService, ContextManager, config

    # Create service
    service = CaptionService(
        session_dir=Path("./session"),
        model=my_model,
        tokenizer=my_tokenizer
    )

    # Start background processing
    service.start()

    # Process signs
    result = service.process_sign(video_bytes)

    # Get caption
    caption = service.get_current_caption()

    # Stop
    service.stop()
"""

from .caption_service import CaptionService
from .context_manager import ContextManager
from . import config

__all__ = ['CaptionService', 'ContextManager', 'config']
__version__ = '1.0.0'
