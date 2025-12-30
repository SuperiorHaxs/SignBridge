import os
import json
import pickle
from collections import Counter

def create_vocab_from_pickle_files():
    """
    Create vocab.json from the pickle files in your dataset
    """
    pickle_files_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/pickle_files"
    output_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete"
    
    # Collect all unique glosses
    glosses = set()
    gloss_counts = Counter()
    
    # Read all pickle files to get glosses
    processed_count = 0
    max_files = 1000  # Limit to first 1000 files
    
    for pkl_file in os.listdir(pickle_files_dir):
        if pkl_file.endswith('.pkl'):
            if processed_count >= max_files:
                break
            try:
                with open(os.path.join(pickle_files_dir, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                    gloss = data['gloss']
                    glosses.add(gloss)
                    gloss_counts[gloss] += 1
                processed_count += 1
            except Exception as e:
                print(f"Error reading {pkl_file}: {e}")
    
    # Sort glosses alphabetically for consistent indexing
    sorted_glosses = sorted(list(glosses))
    
    # Create vocabulary mapping: index -> gloss
    vocab = {}
    for i, gloss in enumerate(sorted_glosses):
        vocab[str(i)] = gloss
    
    # Save vocab.json
    vocab_file = os.path.join(output_dir, 'vocab.json')
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Created vocab.json with {len(vocab)} classes from {processed_count} pickle files")
    print(f"Saved to: {vocab_file}")
    print(f"\nFirst 10 classes:")
    for i in range(min(10, len(vocab))):
        print(f"  {i}: {vocab[str(i)]}")
    
    print(f"\nMost common glosses:")
    for gloss, count in gloss_counts.most_common(10):
        print(f"  {gloss}: {count} samples")
    
    return vocab_file

def create_vocab_from_metadata():
    """
    Alternative: Create vocab.json directly from WLASL metadata
    """
    metadata_file = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/WLASL_v0.3.json"
    output_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete"
    
    # Load WLASL metadata
    with open(metadata_file, 'r') as f:
        wlasl_data = json.load(f)
    
    # Extract all unique glosses
    glosses = set()
    for entry in wlasl_data:
        glosses.add(entry['gloss'])
    
    # Sort glosses alphabetically for consistent indexing
    sorted_glosses = sorted(list(glosses))
    
    # Create vocabulary mapping: index -> gloss
    vocab = {}
    for i, gloss in enumerate(sorted_glosses):
        vocab[str(i)] = gloss
    
    # Save vocab.json
    vocab_file = os.path.join(output_dir, 'vocab.json')
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Created vocab.json with {len(vocab)} classes from metadata")
    print(f"Saved to: {vocab_file}")
    
    return vocab_file

def verify_vocab_with_dataset():
    """
    Verify that the vocab matches your actual dataset
    """
    vocab_file = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/vocab.json"
    pickle_files_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/pickle_files"
    
    # Load vocab
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    
    # Create reverse mapping: gloss -> index
    gloss_to_idx = {gloss: int(idx) for idx, gloss in vocab.items()}
    
    # Check dataset glosses
    dataset_glosses = set()
    missing_glosses = set()
    
    processed_count = 0
    max_files = 1000  # Limit to first 1000 files
    
    for pkl_file in os.listdir(pickle_files_dir):
        if pkl_file.endswith('.pkl'):
            if processed_count >= max_files:
                break
            try:
                with open(os.path.join(pickle_files_dir, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                    gloss = data['gloss']
                    dataset_glosses.add(gloss)
                    
                    if gloss not in gloss_to_idx:
                        missing_glosses.add(gloss)
                processed_count += 1
            except Exception as e:
                print(f"Error reading {pkl_file}: {e}")
    
    print(f"Vocab contains {len(vocab)} glosses")
    print(f"Dataset contains {len(dataset_glosses)} unique glosses from {processed_count} files")
    print(f"Missing from vocab: {len(missing_glosses)}")
    
    if missing_glosses:
        print(f"Missing glosses: {list(missing_glosses)[:10]}")
        return False
    else:
        print("✓ All dataset glosses are in vocab")
        return True

if __name__ == "__main__":
    print("Creating vocab.json for WLASL dataset (first 1000 files)...")
    
    # Method 1: Create from your actual pickle files (recommended)
    vocab_file = create_vocab_from_pickle_files()
    
    # Verify the vocab matches your dataset
    print("\nVerifying vocab...")
    if verify_vocab_with_dataset():
        print(f"\n✓ vocab.json is ready to use: {vocab_file}")
        print("\nTo use in your training script:")
        print(f'vocab_path = r"{vocab_file}"')
    else:
        print("\n⚠ There are issues with the vocab file")
