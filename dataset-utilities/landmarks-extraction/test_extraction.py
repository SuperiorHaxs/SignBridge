#!/usr/bin/env python3
"""
Test Landmark Extraction

Verifies that landmark extraction works correctly on sample data.
Run this script to validate the landmarks-extraction module.

Usage:
    python test_extraction.py
    python test_extraction.py --verbose
    python test_extraction.py --sample-file /path/to/sample.pkl
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add script directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

# Import from local files
from landmark_config import (
    LANDMARK_CONFIGS,
    POSE_INDICES,
    LEFT_HAND_INDICES,
    RIGHT_HAND_INDICES,
    FACE_MINIMAL_INDICES,
)
from extract_landmarks import (
    extract_landmarks,
    extract_landmarks_from_pickle,
    get_landmark_info,
    flatten_landmarks,
    unflatten_landmarks,
)


def test_config_consistency():
    """Test that landmark configurations are internally consistent."""
    print("Testing configuration consistency...")

    for config_name, config in LANDMARK_CONFIGS.items():
        indices = config['indices']
        expected_points = config['total_points']
        breakdown = config['breakdown']

        # Check index count matches expected points
        assert len(indices) == expected_points, \
            f"{config_name}: index count {len(indices)} != expected {expected_points}"

        # Check breakdown sums to total
        breakdown_sum = sum(breakdown.values())
        assert breakdown_sum == expected_points, \
            f"{config_name}: breakdown sum {breakdown_sum} != expected {expected_points}"

        # Check no duplicate indices
        assert len(indices) == len(set(indices)), \
            f"{config_name}: duplicate indices found"

        print(f"  [OK] {config_name}: {expected_points} points, breakdown matches")

    print("  All configurations consistent!\n")
    return True


def test_extraction_shapes():
    """Test extraction produces correct output shapes."""
    print("Testing extraction output shapes...")

    # Create synthetic test data
    frames = 30
    keypoints = 576
    coords = 2

    # Random data with some structure
    np.random.seed(42)
    test_data = np.random.randn(frames, keypoints, coords).astype(np.float32)

    # Add realistic shoulder positions for normalization
    test_data[:, 11, :] = [0.3, 0.3]  # left shoulder
    test_data[:, 12, :] = [0.7, 0.3]  # right shoulder

    for config_name, config in LANDMARK_CONFIGS.items():
        expected_points = config['total_points']

        # Extract without normalization first
        extracted = extract_landmarks(test_data, config=config_name, normalize=False)

        expected_shape = (frames, expected_points, coords)
        assert extracted.shape == expected_shape, \
            f"{config_name}: expected {expected_shape}, got {extracted.shape}"

        # Extract with normalization
        extracted_norm = extract_landmarks(test_data, config=config_name, normalize=True)
        assert extracted_norm.shape == expected_shape, \
            f"{config_name} (normalized): expected {expected_shape}, got {extracted_norm.shape}"

        print(f"  [OK] {config_name}: {expected_shape}")

    print("  All shapes correct!\n")
    return True


def test_extraction_values():
    """Test that extraction picks correct indices."""
    print("Testing extraction picks correct values...")

    # Create data where each keypoint has unique values
    frames = 5
    keypoints = 576
    coords = 2

    # Each keypoint has value [index, index+0.5]
    test_data = np.zeros((frames, keypoints, coords), dtype=np.float32)
    for i in range(keypoints):
        test_data[:, i, 0] = i
        test_data[:, i, 1] = i + 0.5

    # Test 75pt config (no normalization to check raw values)
    extracted = extract_landmarks(test_data, config='75pt', normalize=False)

    # Check pose landmarks (indices 0-32 should be at positions 0-32)
    for i, pose_idx in enumerate(POSE_INDICES):
        expected_x = pose_idx
        actual_x = extracted[0, i, 0]
        assert actual_x == expected_x, \
            f"Pose point {i}: expected x={expected_x}, got {actual_x}"

    # Check left hand landmarks (indices 501-521 should be at positions 33-53)
    for i, hand_idx in enumerate(LEFT_HAND_INDICES):
        pos = 33 + i  # Position in extracted array
        expected_x = hand_idx
        actual_x = extracted[0, pos, 0]
        assert actual_x == expected_x, \
            f"Left hand point {i}: expected x={expected_x}, got {actual_x}"

    # Check right hand landmarks (indices 522-542 should be at positions 54-74)
    for i, hand_idx in enumerate(RIGHT_HAND_INDICES):
        pos = 54 + i  # Position in extracted array
        expected_x = hand_idx
        actual_x = extracted[0, pos, 0]
        assert actual_x == expected_x, \
            f"Right hand point {i}: expected x={expected_x}, got {actual_x}"

    print("  [OK] 75pt extraction picks correct indices")

    # Test 83pt config - check face landmarks are included
    extracted_83 = extract_landmarks(test_data, config='83pt', normalize=False)

    # Face landmarks should be at positions 75-82
    for i, face_idx in enumerate(FACE_MINIMAL_INDICES):
        pos = 75 + i  # Position in extracted array
        expected_x = face_idx
        actual_x = extracted_83[0, pos, 0]
        assert actual_x == expected_x, \
            f"Face point {i}: expected x={expected_x}, got {actual_x}"

    print("  [OK] 83pt extraction includes face landmarks correctly")
    print("  All values correct!\n")
    return True


def test_flatten_unflatten():
    """Test flatten and unflatten are inverses."""
    print("Testing flatten/unflatten roundtrip...")

    # Create test data
    frames = 10
    points = 83
    coords = 2

    original = np.random.randn(frames, points, coords).astype(np.float32)

    # Flatten then unflatten
    flattened = flatten_landmarks(original)
    assert flattened.shape == (frames, points * coords), \
        f"Flattened shape wrong: expected {(frames, points * coords)}, got {flattened.shape}"

    unflattened = unflatten_landmarks(flattened, num_points=points, coords=coords)
    assert unflattened.shape == original.shape, \
        f"Unflattened shape wrong: expected {original.shape}, got {unflattened.shape}"

    assert np.allclose(original, unflattened), \
        "Unflatten did not recover original values"

    print(f"  [OK] Roundtrip preserves shape: {original.shape} -> {flattened.shape} -> {unflattened.shape}")
    print("  Flatten/unflatten working correctly!\n")
    return True


def test_pickle_extraction(sample_path: Path = None, verbose: bool = False):
    """Test extraction from actual pickle file."""
    print("Testing pickle file extraction...")

    if sample_path is None:
        # Find a sample pickle file
        datasets_dir = project_root / "datasets" / "wlasl_poses_complete" / "dataset_splits"
        sample_files = list(datasets_dir.glob("**/train/**/*.pkl"))

        if not sample_files:
            print("  ! No pickle files found in datasets directory, skipping pickle test")
            return True

        sample_path = sample_files[0]

    print(f"  Using sample: {sample_path.name}")

    # Test extraction with different configs
    for config_name in ['75pt', '83pt']:
        extracted, metadata = extract_landmarks_from_pickle(
            sample_path,
            config=config_name,
            normalize=True,
        )

        config_info = LANDMARK_CONFIGS[config_name]
        expected_points = config_info['total_points']

        frames = extracted.shape[0]
        points = extracted.shape[1]
        coords = extracted.shape[2]

        assert points == expected_points, \
            f"{config_name}: expected {expected_points} points, got {points}"

        if verbose:
            print(f"  {config_name}:")
            print(f"    Shape: {extracted.shape}")
            print(f"    Metadata: video_id={metadata['video_id']}, gloss={metadata['gloss']}")
            print(f"    Value range: [{extracted.min():.3f}, {extracted.max():.3f}]")
        else:
            print(f"  [OK] {config_name}: {frames} frames x {points} points x {coords} coords")

    print("  Pickle extraction working!\n")
    return True


def test_get_landmark_info():
    """Test landmark info retrieval."""
    print("Testing landmark info retrieval...")

    for config_name in LANDMARK_CONFIGS.keys():
        info = get_landmark_info(config_name)

        assert 'total_points' in info
        assert 'feature_count_xy' in info
        assert info['feature_count_xy'] == info['total_points'] * 2

        print(f"  [OK] {config_name}: {info['total_points']} points = {info['feature_count_xy']} features (xy)")

    print("  Info retrieval working!\n")
    return True


def run_all_tests(sample_path: Path = None, verbose: bool = False):
    """Run all tests."""
    print("=" * 60)
    print("LANDMARK EXTRACTION MODULE TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Config Consistency", test_config_consistency),
        ("Extraction Shapes", test_extraction_shapes),
        ("Extraction Values", test_extraction_values),
        ("Flatten/Unflatten", test_flatten_unflatten),
        ("Info Retrieval", test_get_landmark_info),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"X {name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"X {name} ERROR: {e}\n")
            import traceback
            traceback.print_exc()

    # Pickle test (may skip if no files available)
    try:
        if test_pickle_extraction(sample_path, verbose):
            passed += 1
    except Exception as e:
        failed += 1
        print(f"X Pickle Extraction ERROR: {e}\n")
        import traceback
        traceback.print_exc()

    # Summary
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n[SUCCESS] All tests passed! Landmark extraction module is ready.\n")
        return True
    else:
        print(f"\nX {failed} test(s) failed. Please check the errors above.\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test landmark extraction module")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    parser.add_argument('--sample-file', type=Path,
                       help='Path to specific pickle file to test')

    args = parser.parse_args()

    success = run_all_tests(
        sample_path=args.sample_file,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)
