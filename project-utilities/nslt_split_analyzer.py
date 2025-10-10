import json
import os
from collections import Counter

def analyze_nslt_splits(nslt_file_path, class_list_file_path):
    """
    Analyze NSLT JSON file and show video IDs by train/test/val splits
    """
    
    # Load class list (index -> gloss mapping)
    print("Loading class list...")
    index_to_gloss = {}
    try:
        with open(class_list_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    index = int(parts[0])
                    gloss = parts[1].strip()
                    index_to_gloss[index] = gloss
        
        print(f"Loaded {len(index_to_gloss)} class mappings")
        
    except Exception as e:
        print(f"Error loading class list: {e}")
        return
    
    # Load NSLT data
    print("Loading NSLT metadata...")
    try:
        with open(nslt_file_path, 'r') as f:
            nslt_data = json.load(f)
        
        print(f"Loaded metadata for {len(nslt_data)} videos")
        
    except Exception as e:
        print(f"Error loading NSLT file: {e}")
        return
    
    # Organize by splits
    splits = {
        'train': [],
        'test': [],
        'val': []
    }
    
    # Count by split and gloss
    split_counts = Counter()
    gloss_by_split = {
        'train': Counter(),
        'test': Counter(),
        'val': Counter()
    }
    
    for video_id, video_info in nslt_data.items():
        subset = video_info.get('subset', 'unknown')
        action_indices = video_info.get('action', [])
        
        # Get primary gloss
        gloss = 'unknown'
        if action_indices and len(action_indices) > 0:
            primary_action_index = action_indices[0]
            if primary_action_index in index_to_gloss:
                gloss = index_to_gloss[primary_action_index]
        
        # Add to appropriate split
        if subset in splits:
            splits[subset].append({
                'video_id': video_id,
                'gloss': gloss,
                'action_indices': action_indices,
                'primary_action': action_indices[0] if action_indices else None
            })
            
            split_counts[subset] += 1
            gloss_by_split[subset][gloss] += 1
        else:
            print(f"Unknown subset '{subset}' for video {video_id}")
    
    # Print summary
    print("\n" + "="*60)
    print("NSLT SPLIT ANALYSIS")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"Total videos: {len(nslt_data)}")
    for split, count in split_counts.items():
        print(f"{split.upper()}: {count} videos")
    
    print(f"\nUnique glosses per split:")
    for split in ['train', 'test', 'val']:
        unique_glosses = len(gloss_by_split[split])
        print(f"{split.upper()}: {unique_glosses} unique glosses")
    
    # Show top glosses per split
    print(f"\nTop 10 glosses per split:")
    for split in ['train', 'test', 'val']:
        print(f"\n{split.upper()}:")
        top_glosses = gloss_by_split[split].most_common(10)
        for gloss, count in top_glosses:
            print(f"  {gloss}: {count} videos")
    
    # Save detailed split files
    output_dir = os.path.dirname(nslt_file_path)
    
    for split_name, videos in splits.items():
        if videos:
            output_file = os.path.join(output_dir, f"{split_name}_videos.txt")
            with open(output_file, 'w') as f:
                f.write(f"NSLT {split_name.upper()} SPLIT\n")
                f.write("="*40 + "\n\n")
                f.write(f"Total videos: {len(videos)}\n")
                f.write(f"Unique glosses: {len(set(v['gloss'] for v in videos))}\n\n")
                f.write("Video ID\tGloss\tAction Index\n")
                f.write("-"*40 + "\n")
                
                for video in sorted(videos, key=lambda x: x['gloss']):
                    f.write(f"{video['video_id']}\t{video['gloss']}\t{video['primary_action']}\n")
            
            print(f"\nSaved {split_name} videos to: {output_file}")
    
    # Create cross-reference file
    cross_ref_file = os.path.join(output_dir, "nslt_video_lookup.txt")
    with open(cross_ref_file, 'w') as f:
        f.write("NSLT VIDEO LOOKUP\n")
        f.write("="*50 + "\n\n")
        f.write("Video ID\tSplit\tGloss\tAction Index\n")
        f.write("-"*50 + "\n")
        
        all_videos = []
        for split_name, videos in splits.items():
            for video in videos:
                all_videos.append({
                    'video_id': video['video_id'],
                    'split': split_name,
                    'gloss': video['gloss'],
                    'action': video['primary_action']
                })
        
        # Sort by video ID
        for video in sorted(all_videos, key=lambda x: x['video_id']):
            f.write(f"{video['video_id']}\t{video['split']}\t{video['gloss']}\t{video['action']}\n")
    
    print(f"Saved complete lookup to: {cross_ref_file}")
    
    return splits, split_counts, gloss_by_split

def check_converted_pickles(pickle_dir, splits):
    """
    Check which of your converted pickle files correspond to which original splits
    """
    print("\n" + "="*60)
    print("CHECKING CONVERTED PICKLE FILES")
    print("="*60)
    
    if not os.path.exists(pickle_dir):
        print(f"Pickle directory not found: {pickle_dir}")
        return
    
    # Create lookup from splits
    video_to_split = {}
    for split_name, videos in splits.items():
        for video in videos:
            video_to_split[video['video_id']] = {
                'split': split_name,
                'gloss': video['gloss']
            }
    
    # Check pickle files
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
    print(f"Found {len(pickle_files)} pickle files")
    
    converted_by_split = Counter()
    converted_glosses = Counter()
    
    print(f"\nConverted files by original split:")
    for pkl_file in sorted(pickle_files):
        video_id = pkl_file[:-4]  # Remove .pkl extension
        
        if video_id in video_to_split:
            split_info = video_to_split[video_id]
            converted_by_split[split_info['split']] += 1
            converted_glosses[split_info['gloss']] += 1
        else:
            converted_by_split['unknown'] += 1
    
    for split, count in converted_by_split.items():
        print(f"  {split.upper()}: {count} files")
    
    print(f"\nTop 10 converted glosses:")
    for gloss, count in converted_glosses.most_common(10):
        print(f"  {gloss}: {count} files")

def main():
    # Configuration - UPDATE THESE PATHS
    nslt_file = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/nslt_300.json"
    class_list_file = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_classes_list.txt"
    pickle_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete/pickle_files"
    
    # Analyze NSLT splits
    splits, split_counts, gloss_by_split = analyze_nslt_splits(nslt_file, class_list_file)
    
    # Check which pickle files you have
    check_converted_pickles(pickle_dir, splits)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Files created:")
    print("- train_videos.txt")
    print("- test_videos.txt") 
    print("- val_videos.txt")
    print("- nslt_video_lookup.txt")

if __name__ == "__main__":
    main()
