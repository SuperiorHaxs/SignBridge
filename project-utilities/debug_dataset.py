import os
import pickle

# Check the directory structure
pose_dir = r"C:\Users\padwe\OneDrive\WLASL-proj\WLASL-openhands\wlasl_poses_pickle"

print("=== Directory Analysis ===")
files = os.listdir(pose_dir)
print(f"Total files: {len(files)}")
print(f"First 10 files: {files[:10]}")
print(f"File extensions: {set([f.split('.')[-1] for f in files if '.' in f])}")

# Check filename patterns
print(f"\n=== Filename Patterns ===")
sample_files = files[:5]
for f in sample_files:
    print(f"File: {f}")
    
    # Try to extract gloss from filename
    if '_' in f:
        parts = f.split('_')
        print(f"  Filename parts: {parts}")
    
    # Try to load a sample file to see its structure
    try:
        with open(os.path.join(pose_dir, f), 'rb') as file:
            data = pickle.load(file)
            print(f"  Data type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
            elif hasattr(data, 'shape'):
                print(f"  Shape: {data.shape}")
            else:
                print(f"  Data preview: {str(data)[:100]}...")
    except Exception as e:
        print(f"  Error loading: {e}")
    print()

# Check if there are subdirectories
print(f"\n=== Subdirectories ===")
subdirs = [d for d in os.listdir(pose_dir) if os.path.isdir(os.path.join(pose_dir, d))]
if subdirs:
    print(f"Found subdirectories: {subdirs}")
    # Check first subdirectory
    first_subdir = os.path.join(pose_dir, subdirs[0])
    subdir_files = os.listdir(first_subdir)
    print(f"Files in {subdirs[0]}: {subdir_files[:5]}")
else:
    print("No subdirectories found")
