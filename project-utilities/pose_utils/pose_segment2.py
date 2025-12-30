#!/usr/bin/env python3
"""
pose_segment2.py
EAF-based Pose Segmentation Tool

Uses EAF annotation files to segment continuous pose files into individual signs.
This approach provides precise, manually-annotated segment boundaries.
"""

import xml.etree.ElementTree as ET
import argparse
import os
import copy
import subprocess
from pathlib import Path

# Pose-format imports
try:
    from pose_format import Pose
    print("SUCCESS: pose_format imported successfully")
except ImportError as e:
    print("ERROR: pose_format library not installed!")
    print("Install with: pip install pose-format")
    raise e


# ============================================================================
# AUTO-DETECTION PRESET CONFIGURATIONS
# ============================================================================
# You can easily modify these preset parameters for different video types

AUTO_DETECT_PRESETS = {
    # Concatenated videos (individual signs joined together with clear gaps)
    "concatenated": {
        "extend_before": 1.0,           # Extend 1.0s before each segment
        "extend_after": 1.0,            # Extend 1.0s after each segment
        "merge_gap_threshold": 0.2,     # Merge segments with gaps < 0.2s
        "merge_short_threshold": 0.5,   # Merge short segments with gaps < 0.5s
        "use_adaptive_extension": True, # Use adaptive extension for short segments
        "fill_gaps_threshold": 1.0,     # Fill gaps larger than 1.0s
        "synthetic_coverage": 0.85      # Use 85% of gap for synthetic segments
    },

    # Continuous videos (natural signing flow with minimal pauses)
    "continuous": {
        "extend_before": 0.3,           # Extend 0.3s before each segment
        "extend_after": 0.3,            # Extend 0.3s after each segment
        "merge_gap_threshold": 0.05,    # Merge segments with gaps < 0.05s (very small)
        "merge_short_threshold": 0.1,   # Merge short segments with gaps < 0.1s (conservative)
        "use_adaptive_extension": False,# No adaptive extension
        "fill_gaps_threshold": 0.8,     # Fill gaps larger than 0.8s
        "synthetic_coverage": 0.75      # Use 75% of gap for synthetic segments
    },

    # Mixed videos (hybrid characteristics)
    "mixed": {
        "extend_before": 0.7,           # Extend 0.7s before each segment
        "extend_after": 0.7,            # Extend 0.7s after each segment
        "merge_gap_threshold": 0.18,    # Merge segments with gaps < 0.18s
        "merge_short_threshold": 0.4,   # Merge short segments with gaps < 0.4s
        "use_adaptive_extension": True, # Use adaptive extension
        "fill_gaps_threshold": 0.9,     # Fill gaps larger than 0.9s
        "synthetic_coverage": 0.8       # Use 80% of gap for synthetic segments
    }
}

# Auto-detection scoring thresholds (you can also adjust these if needed)
AUTO_DETECT_THRESHOLDS = {
    "avg_gap_concatenated": 1.0,       # Average gap > 1.0s indicates concatenated (was 1.5s)
    "max_gap_concatenated": 2.5,       # Max gap > 2.5s indicates concatenated (was 3.0s)
    "large_gap_threshold": 1.5,        # Gaps > 1.5s are considered "large" (was 2.0s)
    "large_gap_ratio_threshold": 0.3,  # >30% large gaps indicates concatenated
    "short_segment_threshold": 0.3,    # Segments < 0.3s are considered "short" (was 0.5s)
    "short_segment_ratio_threshold": 0.6,  # >60% short segments indicates concatenated (was 40%)
    "avg_duration_threshold": 0.4,     # Average duration < 0.4s indicates concatenated (was 0.7s)
    "concatenated_score_threshold": 4, # Score >= 4 = concatenated
    "continuous_score_threshold": 1    # Score <= 1 = continuous
}
# ============================================================================


class EAFPoseSegmenter:
    """Segments pose files using EAF annotation timestamps"""

    def __init__(self):
        self.extend_before = 0.0  # seconds to extend before segment start
        self.extend_after = 0.0   # seconds to extend after segment end
        self.merge_gap_threshold = 0.3  # max gap for merging nearby segments
        self.merge_short_threshold = 0.8  # max gap for merging when one segment is short
        self.use_adaptive_extension = False  # whether to use adaptive extension
        self.fill_gaps_threshold = 1.0  # minimum gap size to fill (seconds)
        self.synthetic_coverage = 0.85  # fraction of gap to use for synthetic segments

    def merge_and_filter_segments(self, annotations):
        """Merge nearby segments and filter out very short ones to get main signs"""

        print("\nPROCESSING: Processing segments to find main signs...")

        # Step 1: Filter out only VERY short segments (likely annotation artifacts)
        min_duration_ms = 100  # 0.1 seconds minimum (very conservative)
        filtered = []

        for ann in annotations:
            if ann['duration'] >= min_duration_ms:
                filtered.append(ann)
            else:
                duration_sec = ann['duration'] / 1000.0
                print(f"  Filtering out very short segment: {duration_sec:.2f}s")

        print(f"After filtering very short segments: {len(filtered)} remaining")

        if len(filtered) == 0:
            print("  No segments left after filtering, using original annotations")
            filtered = annotations

        if len(filtered) == 0:
            return annotations  # Return original if all filtered out

        # Step 2: Only merge segments with VERY small gaps (likely annotation artifacts)
        max_gap_ms = 500  # 0.5 second maximum gap to merge (more conservative)
        merged = []

        if len(filtered) == 0:
            return annotations

        current_segment = filtered[0].copy()

        for i in range(1, len(filtered)):
            next_segment = filtered[i]
            gap = next_segment['start_time'] - current_segment['end_time']
            gap_sec = gap / 1000.0
            current_duration = current_segment['duration'] / 1000.0
            next_duration = next_segment['duration'] / 1000.0

            print(f"  Checking gap between segments: {gap_sec:.2f}s (current: {current_duration:.2f}s, next: {next_duration:.2f}s)")

            # Only merge if gap is very small AND segments are both short
            should_merge = (gap <= max_gap_ms and current_duration < 1.0 and next_duration < 1.0)

            if should_merge:
                # Merge with current segment
                print(f"    MERGE: Merging segments: gap = {gap_sec:.2f}s")
                current_segment['end_time'] = next_segment['end_time']
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
            else:
                # Gap is too large or segments too long, finalize current segment and start new one
                print(f"    KEEP: Keeping separate (gap too large or segments substantial)")
                merged.append(current_segment)
                current_segment = next_segment.copy()

        # Add the last segment
        merged.append(current_segment)

        print(f"After conservative merging: {len(merged)} segments")

        # Step 3: Final pass - only merge segments with tiny gaps (< 0.3 seconds)
        # This is for fragments that are clearly part of the same sign
        final_merged = []
        i = 0
        while i < len(merged):
            current = merged[i].copy()

            # Look ahead to see if next segment should be merged
            while i + 1 < len(merged):
                next_seg = merged[i + 1]
                gap = next_seg['start_time'] - current['end_time']
                gap_sec = gap / 1000.0

                # Only merge if gap is VERY small (< 0.3 seconds)
                # This preserves distinct signs while joining obvious fragments
                should_merge = gap_sec < 0.3

                if should_merge:
                    print(f"  Final merge of fragments with {gap_sec:.2f}s gap")
                    current['end_time'] = next_seg['end_time']
                    current['duration'] = current['end_time'] - current['start_time']
                    i += 1  # Skip the merged segment
                else:
                    break

            final_merged.append(current)
            i += 1

        print(f"Final result: {len(final_merged)} segments")

        # Update segment names
        for i, seg in enumerate(final_merged):
            seg['text'] = f"sign_{i+1:03d}"
            duration_sec = seg['duration'] / 1000.0
            start_sec = seg['start_time'] / 1000.0
            end_sec = seg['end_time'] / 1000.0
            print(f"  Final segment {i+1}: {start_sec:.2f}s - {end_sec:.2f}s ({duration_sec:.2f}s)")

        return final_merged

    def intelligent_eaf_processing(self, annotations):
        """Intelligently process EAF annotations to get proper signs"""

        print("\nINTELLIGENT: Intelligent EAF processing...")

        # Separate by tier
        sign_tier = [ann for ann in annotations if ann.get('tier') == 'SIGN']
        sentence_tier = [ann for ann in annotations if ann.get('tier') == 'SENTENCE']

        print(f"Found {len(sign_tier)} SIGN tier segments and {len(sentence_tier)} SENTENCE tier segments")

        # Strategy: Use SIGN tier segments with intelligent merging
        print(f"  Using SIGN tier segments ({len(sign_tier)} segments) with intelligent merging")

        # Sort SIGN tier segments by start time
        sign_tier.sort(key=lambda x: x['start_time'])

        print("Original SIGN tier segments:")
        for i, seg in enumerate(sign_tier):
            start_sec = seg['start_time'] / 1000.0
            end_sec = seg['end_time'] / 1000.0
            duration_sec = seg['duration'] / 1000.0
            print(f"  {i+1}. {start_sec:.2f}s - {end_sec:.2f}s ({duration_sec:.2f}s)")

        # Auto-detect video type and apply appropriate parameters BEFORE merging
        if getattr(self, 'auto_detect_enabled', False):
            video_type = self.detect_video_type(sign_tier)
            self.apply_preset_parameters(video_type)

        # Merge closely spaced SIGN segments that likely belong to the same sign
        merged_segments = []
        if len(sign_tier) > 0:
            current_segment = sign_tier[0].copy()

            for i in range(1, len(sign_tier)):
                next_segment = sign_tier[i]

                # Calculate gap between segments
                gap_ms = next_segment['start_time'] - current_segment['end_time']
                gap_sec = gap_ms / 1000.0

                # Merge criteria for concatenated videos:
                # 1. Gap is very small (< 0.3 seconds)
                # 2. OR one of the segments is very short (< 0.3 seconds) AND gap < 0.8 seconds
                current_duration = current_segment['duration'] / 1000.0
                next_duration = next_segment['duration'] / 1000.0

                should_merge = (gap_sec < self.merge_gap_threshold) or \
                              ((current_duration < 0.3 or next_duration < 0.3) and gap_sec < self.merge_short_threshold)

                if should_merge:
                    print(f"  Merging segments: gap={gap_sec:.2f}s, durations={current_duration:.2f}s+{next_duration:.2f}s")
                    # Merge by extending current segment to include next segment
                    current_segment['end_time'] = next_segment['end_time']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                    # Keep the text from the first segment
                else:
                    # Gap too large, finalize current segment and start new one
                    merged_segments.append(current_segment)
                    current_segment = next_segment.copy()

            # Add the last segment
            merged_segments.append(current_segment)
        else:
            merged_segments = sign_tier

        print(f"\nAfter intelligent merging: {len(merged_segments)} segments")
        for i, seg in enumerate(merged_segments):
            start_sec = seg['start_time'] / 1000.0
            end_sec = seg['end_time'] / 1000.0
            duration_sec = seg['duration'] / 1000.0
            print(f"  {i+1}. {start_sec:.2f}s - {end_sec:.2f}s ({duration_sec:.2f}s) [MERGED]")

        # Don't do gap filling here - it will be done after extension
        print(f"\nAfter intelligent merging (before extension): {len(merged_segments)} segments")

        return merged_segments

    def detect_video_type(self, sign_segments):
        """Detect if video is concatenated or continuous based on segment patterns"""

        if len(sign_segments) < 2:
            return "continuous"  # Default for very short videos

        # Sort segments by start time
        sorted_segments = sorted(sign_segments, key=lambda x: x['start_time'])

        # Calculate segment durations and gaps
        durations = []
        gaps = []

        for i, seg in enumerate(sorted_segments):
            duration = seg['duration'] / 1000.0
            durations.append(duration)

            if i > 0:
                gap = (seg['start_time'] - sorted_segments[i-1]['end_time']) / 1000.0
                gaps.append(gap)

        if not gaps:
            return "continuous"

        # Calculate statistics
        avg_duration = sum(durations) / len(durations)
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        min_gap = min(gaps)

        # Get thresholds from configuration
        thresholds = AUTO_DETECT_THRESHOLDS

        # Count significant gaps using configurable threshold
        large_gaps = [g for g in gaps if g > thresholds["large_gap_threshold"]]
        large_gap_ratio = len(large_gaps) / len(gaps)

        # Count very short segments using configurable threshold
        short_segments = [d for d in durations if d < thresholds["short_segment_threshold"]]
        short_segment_ratio = len(short_segments) / len(durations)

        print(f"\nANALYSIS: Video Type Detection Analysis:")
        print(f"   Segments: {len(sign_segments)}, Gaps: {len(gaps)}")
        print(f"   Avg segment duration: {avg_duration:.2f}s")
        print(f"   Avg gap: {avg_gap:.2f}s, Max gap: {max_gap:.2f}s")
        print(f"   Large gaps (>{thresholds['large_gap_threshold']}s): {len(large_gaps)}/{len(gaps)} ({large_gap_ratio:.1%})")
        print(f"   Short segments (<{thresholds['short_segment_threshold']}s): {len(short_segments)}/{len(durations)} ({short_segment_ratio:.1%})")

        # Decision logic for concatenated vs continuous using configurable thresholds
        concatenated_score = 0

        # Indicators of concatenated video:
        if avg_gap > thresholds["avg_gap_concatenated"]:  # Large average gaps
            concatenated_score += 2
        if max_gap > thresholds["max_gap_concatenated"]:  # Very large maximum gap
            concatenated_score += 2
        if large_gap_ratio > thresholds["large_gap_ratio_threshold"]:  # High percentage of large gaps
            concatenated_score += 2
        if short_segment_ratio > thresholds["short_segment_ratio_threshold"]:  # High percentage of short segments
            concatenated_score += 1
        if avg_duration < thresholds["avg_duration_threshold"]:  # Very short average segments
            concatenated_score += 1

        # Determine video type using configurable thresholds
        if concatenated_score >= thresholds["concatenated_score_threshold"]:
            video_type = "concatenated"
        elif concatenated_score <= thresholds["continuous_score_threshold"]:
            video_type = "continuous"
        else:
            video_type = "mixed"  # Between the thresholds

        print(f"   Concatenated score: {concatenated_score}/8")
        print(f"   DETECTED: Video type: {video_type.upper()}")

        return video_type

    def apply_preset_parameters(self, video_type):
        """Apply appropriate parameters based on detected video type"""

        print(f"\nAPPLYING: Applying {video_type} video preset parameters...")

        # Get preset configuration from the top-of-file configuration
        if video_type in AUTO_DETECT_PRESETS:
            preset = AUTO_DETECT_PRESETS[video_type]

            # Apply all parameters from preset
            self.extend_before = preset["extend_before"]
            self.extend_after = preset["extend_after"]
            self.merge_gap_threshold = preset["merge_gap_threshold"]
            self.merge_short_threshold = preset["merge_short_threshold"]
            self.use_adaptive_extension = preset["use_adaptive_extension"]
            self.fill_gaps_threshold = preset["fill_gaps_threshold"]
            self.synthetic_coverage = preset["synthetic_coverage"]

            # Display applied settings
            print(f"   Extension: {preset['extend_before']}s before / {preset['extend_after']}s after")
            print(f"   Merging: gaps <{preset['merge_gap_threshold']}s, short gaps <{preset['merge_short_threshold']}s")
            print(f"   Gap filling: >{preset['fill_gaps_threshold']}s gaps with {preset['synthetic_coverage']*100:.0f}% coverage")
            print(f"   Adaptive extension: {'enabled' if preset['use_adaptive_extension'] else 'disabled'}")
        else:
            print(f"   WARNING: Unknown video type '{video_type}', using default parameters")

    def fill_gaps_intelligently(self, merged_segments, sentence_tier):
        """Fill significant gaps between segments using SENTENCE tier or synthetic segments"""

        print(f"\nChecking for gaps between {len(merged_segments)} segments...")

        # Sort segments by start time
        merged_segments.sort(key=lambda x: x['start_time'])
        final_segments = []

        for i in range(len(merged_segments)):
            current_segment = merged_segments[i]
            final_segments.append(current_segment)

            # Check gap to next segment
            if i < len(merged_segments) - 1:
                next_segment = merged_segments[i + 1]
                gap_start = current_segment['end_time']
                gap_end = next_segment['start_time']
                gap_duration = (gap_end - gap_start) / 1000.0

                # If gap is significant (> threshold), try to fill it
                if self.fill_gaps_threshold > 0 and gap_duration > self.fill_gaps_threshold:
                    print(f"  Found significant gap: {gap_start/1000.0:.2f}s - {gap_end/1000.0:.2f}s ({gap_duration:.2f}s)")

                    # First, try to find SENTENCE tier segments in this gap
                    gap_filled = False
                    for sent_seg in sentence_tier:
                        sent_start = sent_seg['start_time']
                        sent_end = sent_seg['end_time']

                        # Check if SENTENCE segment is within the gap
                        if gap_start <= sent_start < gap_end and gap_start < sent_end <= gap_end:
                            # SENTENCE segment is entirely within the gap
                            gap_segment = sent_seg.copy()
                            gap_segment['text'] = f"gap_fill_{len(final_segments)+1:03d}"
                            gap_segment['tier'] = 'SENTENCE_FILL'
                            final_segments.append(gap_segment)

                            sent_duration = (sent_end - sent_start) / 1000.0
                            print(f"    Filled with SENTENCE tier: {sent_start/1000.0:.2f}s - {sent_end/1000.0:.2f}s ({sent_duration:.2f}s)")
                            gap_filled = True
                            break

                    # If no SENTENCE segment found, create synthetic segment
                    if not gap_filled:
                        # Create a synthetic segment that uses most of the available gap
                        # For concatenated videos, we want to capture the full sign
                        buffer_time = 0.2  # 0.2s buffer on each side
                        available_duration = gap_duration - (2 * buffer_time)

                        if available_duration > 0.5:  # Only create if meaningful duration
                            # Use configurable coverage of available gap, but ensure reasonable limits
                            synthetic_duration = min(3.0, max(1.0, available_duration * self.synthetic_coverage))

                            # Center the segment in the gap
                            gap_center = (gap_start + gap_end) / 2
                            synthetic_start = int(gap_center - (synthetic_duration * 1000 / 2))
                            synthetic_end = int(gap_center + (synthetic_duration * 1000 / 2))

                            # Ensure it doesn't overlap with existing segments (smaller buffer)
                            synthetic_start = max(gap_start + int(buffer_time * 1000), synthetic_start)
                            synthetic_end = min(gap_end - int(buffer_time * 1000), synthetic_end)

                        if synthetic_end > synthetic_start:
                            synthetic_segment = {
                                'text': f'synthetic_{len(final_segments)+1:03d}',
                                'start_time': synthetic_start,
                                'end_time': synthetic_end,
                                'duration': synthetic_end - synthetic_start,
                                'tier': 'SYNTHETIC_FILL'
                            }
                            final_segments.append(synthetic_segment)

                            synth_duration = (synthetic_end - synthetic_start) / 1000.0
                            print(f"    Created synthetic segment: {synthetic_start/1000.0:.2f}s - {synthetic_end/1000.0:.2f}s ({synth_duration:.2f}s)")

        # Sort final segments by start time
        final_segments.sort(key=lambda x: x['start_time'])

        print(f"\nAfter gap filling: {len(final_segments)} segments")
        for i, seg in enumerate(final_segments):
            start_sec = seg['start_time'] / 1000.0
            end_sec = seg['end_time'] / 1000.0
            duration_sec = seg['duration'] / 1000.0
            tier = seg.get('tier', 'UNKNOWN')
            print(f"  {i+1}. {start_sec:.2f}s - {end_sec:.2f}s ({duration_sec:.2f}s) [{tier}]")

        return final_segments

    def extend_segments(self, annotations, total_duration_seconds=None):
        """Extend segments by specified amounts before and after"""

        if self.extend_before == 0.0 and self.extend_after == 0.0:
            return annotations  # No extension needed

        print(f"\nEXTENDING:  Extending segments by {self.extend_before}s before and {self.extend_after}s after...")

        extended_annotations = []

        for i, ann in enumerate(annotations):
            original_start = ann['start_time']
            original_end = ann['end_time']

            # Convert to seconds for calculation
            start_sec = original_start / 1000.0
            end_sec = original_end / 1000.0
            original_duration = end_sec - start_sec

            # Adaptive extension: shorter segments get proportionally more extension
            # This helps capture complete signs that were under-annotated
            adaptive_before = self.extend_before
            adaptive_after = self.extend_after

            if self.use_adaptive_extension and original_duration < 0.7:  # Very short original segments
                # Give them extra extension to ensure complete capture
                adaptive_before = self.extend_before * 1.3
                adaptive_after = self.extend_after * 1.3
                print(f"    Adaptive extension for short segment ({original_duration:.2f}s): {adaptive_before:.2f}s before, {adaptive_after:.2f}s after")

            # Apply extensions
            new_start_sec = max(0.0, start_sec - adaptive_before)
            new_end_sec = end_sec + adaptive_after

            # If total duration is known, clamp to it
            if total_duration_seconds is not None:
                new_end_sec = min(total_duration_seconds, new_end_sec)

            # Convert back to milliseconds
            new_start_ms = int(new_start_sec * 1000)
            new_end_ms = int(new_end_sec * 1000)

            # Create extended annotation
            extended_ann = ann.copy()
            extended_ann['start_time'] = new_start_ms
            extended_ann['end_time'] = new_end_ms
            extended_ann['duration'] = new_end_ms - new_start_ms

            # Log the extension
            orig_duration = (original_end - original_start) / 1000.0
            new_duration = extended_ann['duration'] / 1000.0
            print(f"  Segment {i+1}: {start_sec:.2f}s-{end_sec:.2f}s ({orig_duration:.2f}s) -> {new_start_sec:.2f}s-{new_end_sec:.2f}s ({new_duration:.2f}s)")
            print(f"    Extension: -{adaptive_before:.2f}s before, +{adaptive_after:.2f}s after")
            print(f"    Timestamps: {extended_ann['start_time']}ms - {extended_ann['end_time']}ms")

            extended_annotations.append(extended_ann)

        # Check for overlaps after extension and resolve them
        resolved_annotations = []
        for i, ann in enumerate(extended_annotations):
            if i == 0:
                resolved_annotations.append(ann)
                continue

            prev_ann = resolved_annotations[-1]

            # Check for overlap with previous segment
            if ann['start_time'] < prev_ann['end_time']:
                # Resolve overlap by meeting in the middle
                overlap_point = int((prev_ann['end_time'] + ann['start_time']) / 2)

                print(f"  Resolving overlap between segments {i} and {i+1} at {overlap_point/1000.0:.2f}s")
                print(f"    Before: prev({prev_ann['start_time']/1000.0:.2f}s-{prev_ann['end_time']/1000.0:.2f}s), curr({ann['start_time']/1000.0:.2f}s-{ann['end_time']/1000.0:.2f}s)")

                # Adjust previous segment end
                prev_ann['end_time'] = overlap_point
                prev_ann['duration'] = prev_ann['end_time'] - prev_ann['start_time']

                # Adjust current segment start
                ann['start_time'] = overlap_point
                ann['duration'] = ann['end_time'] - ann['start_time']

                print(f"    After:  prev({prev_ann['start_time']/1000.0:.2f}s-{prev_ann['end_time']/1000.0:.2f}s), curr({ann['start_time']/1000.0:.2f}s-{ann['end_time']/1000.0:.2f}s)")

            resolved_annotations.append(ann)

        print(f"  Extended {len(annotations)} segments (resolved {len(extended_annotations) - len(resolved_annotations)} overlaps)")

        return resolved_annotations

    def parse_eaf_file(self, eaf_path):
        """Parse EAF file to extract segment annotations"""

        print(f"PARSING: Parsing EAF file: {eaf_path}")

        try:
            tree = ET.parse(eaf_path)
            root = tree.getroot()

            # Find all time slots
            time_slots = {}
            for time_slot in root.findall('.//TIME_SLOT'):
                slot_id = time_slot.get('TIME_SLOT_ID')
                time_value = int(time_slot.get('TIME_VALUE'))  # Usually in milliseconds
                time_slots[slot_id] = time_value

            print(f"Found {len(time_slots)} time slots")

            # Find all annotations from all tiers
            annotations = []
            annotation_counter = 1

            # Look through all tiers to find annotations
            for tier in root.findall('.//TIER'):
                tier_id = tier.get('TIER_ID', 'unknown')
                print(f"Processing tier: {tier_id}")

                for annotation in tier.findall('.//ANNOTATION'):
                    alignable_annotation = annotation.find('ALIGNABLE_ANNOTATION')
                    if alignable_annotation is not None:
                        start_slot = alignable_annotation.get('TIME_SLOT_REF1')
                        end_slot = alignable_annotation.get('TIME_SLOT_REF2')
                        annotation_id = alignable_annotation.get('ANNOTATION_ID', f'a{annotation_counter}')

                        # Get annotation value (gloss/text)
                        annotation_value = alignable_annotation.find('ANNOTATION_VALUE')
                        text = None
                        if annotation_value is not None and annotation_value.text is not None:
                            text = annotation_value.text.strip()

                        # If no text, create a generic name (just number)
                        if not text:
                            text = f"{annotation_counter:03d}"

                        if start_slot in time_slots and end_slot in time_slots:
                            start_time = time_slots[start_slot]
                            end_time = time_slots[end_slot]

                            # Only add if it has a reasonable duration
                            duration = end_time - start_time
                            if duration > 0:  # Must have positive duration
                                annotations.append({
                                    'text': text,
                                    'start_time': start_time,  # milliseconds
                                    'end_time': end_time,      # milliseconds
                                    'duration': duration,
                                    'tier': tier_id,
                                    'annotation_id': annotation_id
                                })
                                annotation_counter += 1

            # Remove duplicates (same start/end times from different tiers)
            unique_annotations = []
            seen_times = set()

            for ann in annotations:
                time_key = (ann['start_time'], ann['end_time'])
                if time_key not in seen_times:
                    unique_annotations.append(ann)
                    seen_times.add(time_key)
                else:
                    print(f"  Skipping duplicate time range: {time_key}")

            annotations = unique_annotations

            # Sort annotations by start time
            annotations.sort(key=lambda x: x['start_time'])

            print(f"Found {len(annotations)} unique annotations:")
            for i, ann in enumerate(annotations):
                start_sec = ann['start_time'] / 1000.0
                end_sec = ann['end_time'] / 1000.0
                duration_sec = ann['duration'] / 1000.0
                tier_info = f"[{ann['tier']}]" if 'tier' in ann else ""
                print(f"  {i+1:2d}. '{ann['text']}' {tier_info} ({start_sec:.2f}s - {end_sec:.2f}s, {duration_sec:.2f}s)")

            # Post-process annotations based on mode
            if getattr(self, 'use_raw_segments', False):
                print(f"\nRAW: Raw mode enabled - using annotations as-is without processing")
                # Just update the segment names for consistency
                for i, ann in enumerate(annotations):
                    ann['text'] = f"sign_{i+1:03d}"
                final_annotations = annotations
            elif getattr(self, 'use_intelligent_processing', False):
                final_annotations = self.intelligent_eaf_processing(annotations)
                # Store sentence tier for gap filling after extension
                sentence_tier = [ann for ann in annotations if ann.get('tier') == 'SENTENCE']
                self._sentence_tier_for_gap_filling = sentence_tier
            else:
                processed_annotations = self.merge_and_filter_segments(annotations)

                print(f"\nAfter merging and filtering: {len(processed_annotations)} main segments:")
                for i, ann in enumerate(processed_annotations):
                    start_sec = ann['start_time'] / 1000.0
                    end_sec = ann['end_time'] / 1000.0
                    duration_sec = ann['duration'] / 1000.0
                    print(f"  {i+1:2d}. '{ann['text']}' ({start_sec:.2f}s - {end_sec:.2f}s, {duration_sec:.2f}s)")

                final_annotations = processed_annotations

            # Apply segment extension if requested
            if hasattr(self, 'extend_before') and hasattr(self, 'extend_after'):
                # We'll apply extension later when we have the pose file duration
                pass

            return final_annotations

        except Exception as e:
            print(f"ERROR: Error parsing EAF file: {e}")
            return []

    def load_pose_file(self, pose_path):
        """Load pose file"""

        print(f"LOADING: Loading pose file: {pose_path}")

        try:
            with open(pose_path, "rb") as f:
                buffer = f.read()
                pose = Pose.read(buffer)

            print(f"SUCCESS: Loaded pose file successfully")
            print(f"   Shape: {pose.body.data.shape}")
            print(f"   FPS: {pose.body.fps}")
            print(f"   Total frames: {pose.body.data.shape[0]}")

            # Calculate duration
            total_frames = pose.body.data.shape[0]
            fps = pose.body.fps
            duration_seconds = total_frames / fps if fps > 0 else 0
            print(f"   Duration: {duration_seconds:.2f} seconds")

            return pose

        except Exception as e:
            print(f"ERROR: Error loading pose file: {e}")
            return None

    def convert_time_to_frames(self, time_ms, fps):
        """Convert milliseconds to frame indices"""
        time_seconds = time_ms / 1000.0
        frame_index = round(time_seconds * fps)
        return frame_index

    def extract_segments(self, pose, annotations, output_dir):
        """Extract pose segments based on EAF annotations"""

        print(f"\nEXTRACTING: Extracting segments to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        fps = pose.body.fps
        total_frames = pose.body.data.shape[0]

        saved_files = []

        for i, annotation in enumerate(annotations):
            try:
                # Convert timestamps to frame indices
                start_frame = self.convert_time_to_frames(annotation['start_time'], fps)
                end_frame = self.convert_time_to_frames(annotation['end_time'], fps)

                # Ensure frames are within bounds
                start_frame = max(0, start_frame)
                end_frame = min(total_frames - 1, end_frame)

                if start_frame >= end_frame:
                    print(f"WARNING:  Skipping invalid segment {i+1}: start >= end")
                    continue

                duration_frames = end_frame - start_frame + 1
                duration_seconds = duration_frames / fps

                # Expected duration from annotation
                expected_duration = (annotation['end_time'] - annotation['start_time']) / 1000.0

                print(f"Segment {i+1}: '{annotation['text']}'")
                print(f"  Time: {annotation['start_time']/1000:.2f}s - {annotation['end_time']/1000:.2f}s")
                print(f"  Expected duration: {expected_duration:.2f}s")
                print(f"  Frames: {start_frame} - {end_frame} ({duration_frames} frames, {duration_seconds:.2f}s)")

                # Check if duration matches expectation
                duration_diff = abs(duration_seconds - expected_duration)
                if duration_diff > 0.1:  # More than 0.1s difference
                    print(f"  WARNING: Duration mismatch! Expected {expected_duration:.2f}s, got {duration_seconds:.2f}s (diff: {duration_diff:.2f}s)")

                # Extract segment data
                segment_data = pose.body.data[start_frame:end_frame+1]
                segment_confidence = None
                if hasattr(pose.body, 'confidence') and pose.body.confidence is not None:
                    segment_confidence = pose.body.confidence[start_frame:end_frame+1]

                # Create new pose header for segment
                segment_header = copy.deepcopy(pose.header)

                # Update frame count in header if possible
                if hasattr(segment_header, "frames"):
                    segment_header.frames = segment_data.shape[0]
                elif hasattr(segment_header, "length"):
                    segment_header.length = segment_data.shape[0]

                # Create new pose object for segment
                segmented_pose = Pose(
                    header=segment_header,
                    body=type(pose.body)(
                        data=segment_data,
                        confidence=segment_confidence,
                        fps=pose.body.fps
                    )
                )

                # Create safe filename from annotation text
                safe_text = "".join(c if c.isalnum() or c in ' -_' else '' for c in annotation['text'])
                safe_text = safe_text.replace(' ', '_')[:50]  # Limit length

                # If the text is already a number (like "001"), don't add "segment_" prefix
                if safe_text.isdigit():
                    output_filename = f"sign_{safe_text}.pose"
                else:
                    output_filename = f"segment_{i+1:03d}_{safe_text}.pose"
                output_path = os.path.join(output_dir, output_filename)

                # Save segment
                with open(output_path, "wb") as f:
                    segmented_pose.write(f)

                saved_files.append(output_path)
                print(f"SUCCESS: Saved: {output_path}")

            except Exception as e:
                print(f"ERROR: Error processing segment {i+1}: {e}")

        print(f"\nSUCCESS: Successfully extracted {len(saved_files)} segments")
        return saved_files

    def create_visualizations(self, saved_files, output_dir):
        """Create MP4 visualizations for each pose segment"""

        print(f"\nVISUALIZING: Creating visualizations...")

        video_files = []

        for pose_file in saved_files:
            try:
                # Get the base name without extension
                base_name = os.path.splitext(os.path.basename(pose_file))[0]
                video_path = os.path.join(output_dir, f"viz_{base_name}.mp4")

                # Call visualize_pose executable
                cmd = ["visualize_pose", "-i", pose_file, "-o", video_path, "--normalize"]
                print(f"Running: visualize_pose -i {os.path.basename(pose_file)} -o {os.path.basename(video_path)}")

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    video_files.append(video_path)
                    print(f"SUCCESS: Created: {os.path.basename(video_path)}")
                else:
                    print(f"ERROR: Failed to create video for {os.path.basename(pose_file)}")
                    if result.stderr:
                        print(f"   Error: {result.stderr}")

            except Exception as e:
                print(f"ERROR: Error creating video for {os.path.basename(pose_file)}: {e}")

        print(f"\nSUCCESS: Created {len(video_files)} visualization videos")
        return video_files

    def process_eaf_pose_pair(self, eaf_path, pose_path, output_dir):
        """Complete pipeline: EAF + Pose -> Segmented poses + Videos"""

        print("PIPELINE: EAF-based Pose Segmentation Pipeline")
        print("="*50)

        # Parse EAF file
        annotations = self.parse_eaf_file(eaf_path)
        if not annotations:
            print("ERROR: No annotations found in EAF file")
            return None, None

        # Load pose file
        pose = self.load_pose_file(pose_path)
        if pose is None:
            print("ERROR: Failed to load pose file")
            return None, None

        # Calculate total duration for extension boundary checking
        total_frames = pose.body.data.shape[0]
        fps = pose.body.fps
        total_duration_seconds = total_frames / fps if fps > 0 else None

        # Apply segment extension if requested (now that we have pose duration)
        if hasattr(self, 'extend_before') and hasattr(self, 'extend_after') and (self.extend_before > 0 or self.extend_after > 0):
            annotations = self.extend_segments(annotations, total_duration_seconds)

        # After extension, check for gaps and fill them if needed
        if (hasattr(self, 'use_intelligent_processing') and self.use_intelligent_processing and
            hasattr(self, 'fill_gaps_threshold') and self.fill_gaps_threshold > 0):
            sentence_tier = getattr(self, '_sentence_tier_for_gap_filling', [])
            annotations = self.fill_gaps_intelligently(annotations, sentence_tier)

        # Extract segments
        saved_files = self.extract_segments(pose, annotations, output_dir)
        if not saved_files:
            print("ERROR: No segments extracted")
            return None, None

        # Create visualizations
        video_files = self.create_visualizations(saved_files, output_dir)

        # Summary
        print("\n" + "="*50)
        print("SUMMARY: EXTRACTION SUMMARY:")
        print(f"   EAF file: {os.path.basename(eaf_path)}")
        print(f"   Pose file: {os.path.basename(pose_path)}")
        print(f"   Annotations: {len(annotations)}")
        print(f"   Pose segments: {len(saved_files)}")
        print(f"   Visualization videos: {len(video_files)}")
        print(f"   Output directory: {output_dir}")

        print(f"\nFILES: Generated files:")
        for pose_file in saved_files:
            base_name = os.path.basename(pose_file)
            video_name = f"viz_{os.path.splitext(base_name)[0]}.mp4"
            print(f"   {base_name} + {video_name}")

        return saved_files, video_files


def main():
    """Main function"""

    parser = argparse.ArgumentParser(
        description="EAF-based Pose Segmentation Tool",
        epilog="""
Examples:
  # Basic usage
  python pose_segment2.py video.eaf video.pose

  # Extend segments by 0.5 seconds on each side for better sign coverage
  python pose_segment2.py video.eaf video.pose --extend-segments 0.5

  # Use intelligent processing with merging of close segments
  python pose_segment2.py video.eaf video.pose --intelligent --extend-segments 1.0

  # Merge segments with gaps < 0.2s, and use adaptive extension
  python pose_segment2.py video.eaf video.pose --intelligent --extend-segments 1.0 --merge-gap 0.2 --adaptive-extension

  # Custom extension: more before start, less after end
  python pose_segment2.py video.eaf video.pose --extend-before 0.7 --extend-after 0.3

  # For concatenated videos: aggressive merging + gap filling
  python pose_segment2.py video.eaf video.pose --intelligent --extend-segments 1.0 --merge-gap 0.15 --merge-short 0.5 --fill-gaps 1.0

  # Disable gap filling if you only want explicit annotations
  python pose_segment2.py video.eaf video.pose --intelligent --extend-segments 1.0 --fill-gaps 0

  # Make synthetic segments use more of the available gap (90% instead of 85%)
  python pose_segment2.py video.eaf video.pose --intelligent --extend-segments 1.0 --synthetic-coverage 0.9

  # Automatically detect video type and apply appropriate parameters
  python pose_segment2.py video.eaf video.pose --intelligent --auto-detect
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("eaf_file", help="Path to the .eaf annotation file")
    parser.add_argument("pose_file", help="Path to the .pose file")
    parser.add_argument("-o", "--output-dir", default="eaf_segments",
                        help="Output directory for segments (default: eaf_segments)")
    parser.add_argument("--no-videos", action="store_true",
                        help="Skip video visualization creation")
    parser.add_argument("--raw", action="store_true",
                        help="Use raw EAF segments without merging/filtering")
    parser.add_argument("--intelligent", action="store_true",
                        help="Use intelligent processing (SENTENCE tier + gap filling)")
    parser.add_argument("--extend-segments", type=float, default=0.0,
                        help="Extend segments by N seconds on each side (e.g., 0.5 for 0.5s before/after)")
    parser.add_argument("--extend-before", type=float, default=None,
                        help="Extend segments by N seconds before start (overrides --extend-segments)")
    parser.add_argument("--extend-after", type=float, default=None,
                        help="Extend segments by N seconds after end (overrides --extend-segments)")
    parser.add_argument("--merge-gap", type=float, default=0.3,
                        help="Maximum gap (seconds) for merging nearby segments (default: 0.3)")
    parser.add_argument("--merge-short", type=float, default=0.8,
                        help="Maximum gap (seconds) for merging when one segment is short (default: 0.8)")
    parser.add_argument("--adaptive-extension", action="store_true",
                        help="Use adaptive extension (more extension for shorter segments)")
    parser.add_argument("--fill-gaps", type=float, default=1.0,
                        help="Fill gaps larger than N seconds with SENTENCE tier or synthetic segments (default: 1.0, 0 to disable)")
    parser.add_argument("--synthetic-coverage", type=float, default=0.85,
                        help="Fraction of gap to use for synthetic segments (0.5-1.0, default: 0.85)")
    parser.add_argument("--auto-detect", action="store_true",
                        help="Automatically detect video type (concatenated vs continuous) and apply appropriate parameters")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.eaf_file):
        print(f"ERROR: EAF file not found: {args.eaf_file}")
        return

    if not os.path.exists(args.pose_file):
        print(f"ERROR: Pose file not found: {args.pose_file}")
        return

    # Create segmenter
    segmenter = EAFPoseSegmenter()
    segmenter.use_raw_segments = args.raw
    segmenter.use_intelligent_processing = args.intelligent

    # Handle auto-detection if requested - store annotations for reuse
    initial_annotations = None
    if args.auto_detect:
        print("\nAUTO-DETECT: Analyzing video type...")
        # Parse EAF first to analyze segment patterns
        initial_annotations = segmenter.parse_eaf_file(args.eaf_file)
        if initial_annotations:
            # Filter to only SIGN tier for analysis
            sign_segments = [ann for ann in initial_annotations if ann.get('tier') == 'SIGN']
            video_type = segmenter.detect_video_type(sign_segments)
            print(f"DETECTED: Video type detected as: {video_type}")
            segmenter.apply_preset_parameters(video_type)
            print(f"APPLIED: Applied {video_type} preset parameters")
            # Enable auto-detection flag for subsequent processing
            segmenter.auto_detect_enabled = True
        else:
            print("WARNING: Could not analyze EAF for auto-detection, using default parameters")

    # Set extension parameters (only if not using auto-detection)
    if not args.auto_detect:
        if args.extend_before is not None:
            segmenter.extend_before = args.extend_before
        else:
            segmenter.extend_before = args.extend_segments

        if args.extend_after is not None:
            segmenter.extend_after = args.extend_after
        else:
            segmenter.extend_after = args.extend_segments

        # Set merging and adaptive extension parameters
        segmenter.merge_gap_threshold = args.merge_gap
        segmenter.merge_short_threshold = args.merge_short
        segmenter.use_adaptive_extension = args.adaptive_extension
        segmenter.fill_gaps_threshold = args.fill_gaps
        segmenter.synthetic_coverage = args.synthetic_coverage
    else:
        # When using auto-detection, allow manual parameters to override preset if explicitly provided
        if args.extend_before is not None:
            segmenter.extend_before = args.extend_before
            print(f"OVERRIDE: Manual extend_before: {args.extend_before}s")
        if args.extend_after is not None:
            segmenter.extend_after = args.extend_after
            print(f"OVERRIDE: Manual extend_after: {args.extend_after}s")
        # For other parameters, only override if they differ from defaults
        if args.merge_gap != 0.3:  # default value
            segmenter.merge_gap_threshold = args.merge_gap
            print(f"OVERRIDE: Manual merge_gap: {args.merge_gap}s")
        if args.merge_short != 0.8:  # default value
            segmenter.merge_short_threshold = args.merge_short
            print(f"OVERRIDE: Manual merge_short: {args.merge_short}s")
        if args.adaptive_extension:
            segmenter.use_adaptive_extension = args.adaptive_extension
            print(f"OVERRIDE: Manual adaptive_extension: {args.adaptive_extension}")
        if args.fill_gaps != 1.0:  # default value
            segmenter.fill_gaps_threshold = args.fill_gaps
            print(f"OVERRIDE: Manual fill_gaps: {args.fill_gaps}s")
        if args.synthetic_coverage != 0.85:  # default value
            segmenter.synthetic_coverage = args.synthetic_coverage
            print(f"OVERRIDE: Manual synthetic_coverage: {args.synthetic_coverage}")

    # Process files
    if args.no_videos:
        # Parse EAF and extract segments only (reuse if auto-detection already parsed)
        annotations = initial_annotations if initial_annotations is not None else segmenter.parse_eaf_file(args.eaf_file)
        if annotations:
            pose = segmenter.load_pose_file(args.pose_file)
            if pose:
                # Calculate total duration for extension boundary checking
                total_frames = pose.body.data.shape[0]
                fps = pose.body.fps
                total_duration_seconds = total_frames / fps if fps > 0 else None

                # Apply segment extension if requested
                if hasattr(segmenter, 'extend_before') and hasattr(segmenter, 'extend_after') and (segmenter.extend_before > 0 or segmenter.extend_after > 0):
                    annotations = segmenter.extend_segments(annotations, total_duration_seconds)

                # After extension, check for gaps and fill them if needed
                if (hasattr(segmenter, 'use_intelligent_processing') and segmenter.use_intelligent_processing and
                    hasattr(segmenter, 'fill_gaps_threshold') and segmenter.fill_gaps_threshold > 0):
                    sentence_tier = getattr(segmenter, '_sentence_tier_for_gap_filling', [])
                    annotations = segmenter.fill_gaps_intelligently(annotations, sentence_tier)

                saved_files = segmenter.extract_segments(pose, annotations, args.output_dir)
                print(f"\nSUCCESS: Extracted {len(saved_files)} pose segments (no videos)")
    else:
        # Full pipeline with videos
        saved_files, video_files = segmenter.process_eaf_pose_pair(
            args.eaf_file,
            args.pose_file,
            args.output_dir
        )

        if saved_files and video_files:
            print(f"\nSUCCESS: Success! Generated {len(saved_files)} pose segments with visualizations")
            print(f"DIRECTORY: Check output directory: {args.output_dir}")


if __name__ == "__main__":
    main()