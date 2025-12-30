"""
Segmentation utilities for ASL video/pose processing.

Provides time-based and motion-based segmentation for webcam recordings.
"""

from .webcam_time_segmenter import TimeBasedSegmenter
from .webcam_motion_segmenter import MotionBasedSegmenter

__all__ = ['TimeBasedSegmenter', 'MotionBasedSegmenter']
