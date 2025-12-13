"""
Camera Diagnostic Tool
Tests available cameras and backends to find working configuration
"""

import cv2
import sys

def test_camera(index, backend=cv2.CAP_ANY):
    """Test if a camera at given index works"""
    backend_names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_DSHOW: "CAP_DSHOW (DirectShow)",
        cv2.CAP_MSMF: "CAP_MSMF (Microsoft Media Foundation)",
        cv2.CAP_VFW: "CAP_VFW (Video for Windows)"
    }

    backend_name = backend_names.get(backend, f"Backend {backend}")

    print(f"\nTesting Camera {index} with {backend_name}...")

    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        print(f"  ✗ Failed to open camera {index}")
        return False

    print(f"  ✓ Camera {index} opened successfully")

    # Try to read a frame
    ret, frame = cap.read()

    if not ret:
        print(f"  ✗ Camera opened but cannot read frames")
        cap.release()
        return False

    print(f"  ✓ Successfully read frame")
    print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"  FPS: {fps}")
    print(f"  Resolution: {int(width)}x{int(height)}")

    cap.release()
    return True

def main():
    print("="*70)
    print("CAMERA DIAGNOSTIC TOOL")
    print("="*70)
    print("Scanning for available cameras...\n")

    backends = [
        cv2.CAP_DSHOW,      # DirectShow (Windows)
        cv2.CAP_MSMF,       # Microsoft Media Foundation (Windows)
        cv2.CAP_ANY         # Auto-detect
    ]

    working_configs = []

    # Test camera indices 0-3
    for camera_index in range(4):
        found = False
        for backend in backends:
            if test_camera(camera_index, backend):
                working_configs.append((camera_index, backend))
                found = True
                break  # Found working backend for this camera

        if not found:
            print(f"\n  Camera {camera_index}: Not available")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if working_configs:
        print(f"\nFound {len(working_configs)} working camera(s):\n")

        backend_names = {
            cv2.CAP_ANY: "CAP_ANY",
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation"
        }

        for idx, (camera_idx, backend) in enumerate(working_configs, 1):
            backend_name = backend_names.get(backend, f"Backend {backend}")
            print(f"  {idx}. Camera {camera_idx} ({backend_name})")
            print(f"     Use: python frontend_cv_service.py --camera-index {camera_idx}")

        print("\nRECOMMENDED COMMAND:")
        print(f"  python frontend_cv_service.py --camera-index {working_configs[0][0]}")

    else:
        print("\n✗ No working cameras found!")
        print("\nTroubleshooting steps:")
        print("  1. Check if another application is using the camera")
        print("  2. Check Windows Privacy Settings > Camera")
        print("  3. Make sure camera drivers are installed")
        print("  4. Try unplugging and replugging USB camera")
        print("  5. Restart your computer")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
