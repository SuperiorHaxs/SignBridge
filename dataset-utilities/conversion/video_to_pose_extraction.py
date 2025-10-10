import os
import json
import subprocess
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pose_format import Pose

def main():
    # Configuration - KEEPING YOUR EXACT PATHS
    video_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/videos"
    metadata_file = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/nslt_2000.json"  # CHANGED TO NSLT FORMAT
    class_list_file = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_class_list.txt"  # NEW FILE
    output_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle/wlasl_poses_complete"
    
    # Create output directories
    pose_files_dir = os.path.join(output_dir, "pose_files")
    pickle_files_dir = os.path.join(output_dir, "pickle_files")
    
    os.makedirs(pose_files_dir, exist_ok=True)
    os.makedirs(pickle_files_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Step 1: Extracting poses from videos (Windows-safe) ===")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).rglob(f'*{ext}'))
    
    print(f"Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("No video files found! Check your video directory path.")
        return
    
    # Extract poses one by one (Windows-safe)
    successful_poses = 0
    failed_poses = []
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            # Create output pose filename
            pose_filename = video_file.stem + '.pose'
            pose_output = os.path.join(pose_files_dir, pose_filename)
            
            # Skip if pose file already exists
            if os.path.exists(pose_output):
                successful_poses += 1
                continue
            
            # Run video_to_pose on single file
            cmd = [
                'video_to_pose',
                '--format', 'mediapipe',
                '-i', str(video_file),
                '-o', pose_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(pose_output):
                successful_poses += 1
            else:
                failed_poses.append(str(video_file))
                
        except Exception as e:
            failed_poses.append(str(video_file))
            print(f"Exception processing {video_file.name}: {e}")
    
    print(f"Pose extraction complete: {successful_poses} successful, {len(failed_poses)} failed")
    
    # Step 2: Load NSLT metadata and class list - COMPLETELY REWRITTEN
    print("=== Step 2: Loading NSLT metadata and class list ===")
    
    # Load class list (index -> gloss mapping)
    index_to_gloss = {}
    try:
        with open(class_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    index = int(parts[0])
                    gloss = parts[1].strip()
                    index_to_gloss[index] = gloss
        
        print(f"Loaded {len(index_to_gloss)} class mappings")
        print(f"Sample mappings: {dict(list(index_to_gloss.items())[:5])}")
        
    except Exception as e:
        print(f"Error loading class list: {e}")
        return
    
    # Load NSLT metadata (video_id -> action indices)
    video_to_gloss = {}
    try:
        with open(metadata_file, 'r') as f:
            nslt_data = json.load(f)
        
        # Convert NSLT format to our format
        for video_id, video_info in nslt_data.items():
            subset = video_info.get('subset', 'train')
            action_indices = video_info.get('action', [])
            
            # Take the first action index as the primary class
            # Note: NSLT might have multiple actions, but we'll use the first one
            if action_indices and len(action_indices) > 0:
                primary_action_index = action_indices[0]
                
                if primary_action_index in index_to_gloss:
                    gloss = index_to_gloss[primary_action_index]
                    
                    video_to_gloss[video_id] = {
                        'gloss': gloss,
                        'split': subset,
                        'action_indices': action_indices,
                        'primary_action': primary_action_index
                    }
        
        print(f"Loaded metadata for {len(video_to_gloss)} videos")
        print(f"Covering {len(set(v['gloss'] for v in video_to_gloss.values()))} unique glosses")
        
        # Show sample mappings
        sample_videos = list(video_to_gloss.items())[:5]
        for vid_id, info in sample_videos:
            print(f"  {vid_id} -> {info['gloss']} (action {info['primary_action']}, {info['split']})")
            
    except Exception as e:
        print(f"Error loading NSLT metadata: {e}")
        return
    
    # Step 3: Convert pose files to pickle format - UPDATED MATCHING LOGIC
    print("=== Step 3: Converting .pose files to pickle format ===")
    
    # Find all .pose files
    pose_files = []
    for root, dirs, files in os.walk(pose_files_dir):
        for file in files:
            if file.endswith('.pose'):
                pose_files.append(os.path.join(root, file))
    
    print(f"Found {len(pose_files)} .pose files")
    
    if len(pose_files) == 0:
        print("No .pose files found! Check if pose extraction worked correctly.")
        return
    
    successful_conversions = 0
    failed_conversions = []
    
    for pose_file in tqdm(pose_files, desc="Converting pose files"):
        try:
            # Extract filename without extension
            pose_filename = Path(pose_file).stem
            
            # Try to match with video IDs using improved logic
            video_id = None
            gloss_info = None
            
            # Direct match
            if pose_filename in video_to_gloss:
                video_id = pose_filename
                gloss_info = video_to_gloss[video_id]
            else:
                # Try zero-padded versions (common with NSLT format)
                for padding in [4, 5, 6]:
                    padded_name = pose_filename.zfill(padding)
                    if padded_name in video_to_gloss:
                        video_id = padded_name
                        gloss_info = video_to_gloss[video_id]
                        break
                
                # Try removing leading zeros
                if video_id is None:
                    clean_name = pose_filename.lstrip('0')
                    if clean_name in video_to_gloss:
                        video_id = clean_name
                        gloss_info = video_to_gloss[video_id]
                
                # Try adding leading zeros to make it 5 digits
                if video_id is None and pose_filename.isdigit():
                    padded_name = pose_filename.zfill(5)
                    if padded_name in video_to_gloss:
                        video_id = padded_name
                        gloss_info = video_to_gloss[video_id]
            
            if video_id is None or gloss_info is None:
                failed_conversions.append(f"{pose_filename} (no metadata match)")
                continue
            
            # Load and convert pose file
            with open(pose_file, 'rb') as f:
                pose = Pose.read(f.read())
            
            keypoints, confidences = pose_to_numpy(pose)
            
            if keypoints is not None:
                # Create pickle data with NSLT information
                pickle_data = {
                    'keypoints': keypoints,
                    'confidences': confidences,
                    'video_id': video_id,
                    'gloss': gloss_info['gloss'],
                    'split': gloss_info['split'],
                    'action_indices': gloss_info['action_indices'],
                    'primary_action': gloss_info['primary_action'],
                    'pose_file': pose_file
                }
                
                # Save pickle file using the matched video_id
                output_filename = f"{video_id}.pkl"
                output_path = os.path.join(pickle_files_dir, output_filename)
                
                with open(output_path, 'wb') as f:
                    pickle.dump(pickle_data, f)
                
                successful_conversions += 1
            else:
                failed_conversions.append(f"{pose_filename} (pose conversion failed)")
                
        except Exception as e:
            print(f"Error processing {pose_file}: {e}")
            failed_conversions.append(f"{pose_filename} (exception: {str(e)})")
    
    # Step 4: Save mapping and create summary - UPDATED
    print("=== Step 4: Saving results ===")
    
    mapping_file = os.path.join(output_dir, 'video_to_gloss_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(video_to_gloss, f, indent=2)
    
    # Also save the index to gloss mapping for reference
    class_mapping_file = os.path.join(output_dir, 'class_index_mapping.json')
    with open(class_mapping_file, 'w') as f:
        json.dump(index_to_gloss, f, indent=2)
    
    # Create summary
    summary_file = os.path.join(output_dir, 'README.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("WLASL Pose Extraction Results (NSLT Format)\n")
        f.write("=" * 50 + "\n\n")
        f.write("Directory Structure:\n")
        f.write("|- pose_files/     - Original .pose files (for visualization)\n")
        f.write("|- pickle_files/   - Pickle files (for training)\n")
        f.write("|- video_to_gloss_mapping.json - Video ID to gloss mapping\n")
        f.write("|- class_index_mapping.json - Class index to gloss mapping\n")
        f.write("L- README.txt      - This file\n\n")
        f.write("Statistics:\n")
        f.write(f"Pose extraction: {successful_poses} successful, {len(failed_poses)} failed\n")
        f.write(f"Pickle conversion: {successful_conversions} successful, {len(failed_conversions)} failed\n")
        if successful_conversions > 0:
            success_rate = successful_conversions/(successful_conversions+len(failed_conversions))*100
            f.write(f"Conversion success rate: {success_rate:.1f}%\n\n")
        
        # Gloss distribution
        gloss_counts = {}
        for info in video_to_gloss.values():
            gloss = info['gloss']
            gloss_counts[gloss] = gloss_counts.get(gloss, 0) + 1
        
        f.write(f"Gloss distribution (top 10):\n")
        sorted_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
        for gloss, count in sorted_glosses[:10]:
            f.write(f"  {gloss}: {count} videos\n")
        
        f.write("\nUsage:\n")
        f.write("- Use .pose files in pose_files/ with your pose visualizer\n")
        f.write("- Use .pkl files in pickle_files/ for training your model\n")
        f.write("- Update your training script to load poses from pickle_files/\n")
    
    print(f"\n=== Extraction Complete ===")
    print(f"Pose extraction: {successful_poses} successful, {len(failed_poses)} failed")
    print(f"Pickle conversion: {successful_conversions} successful, {len(failed_conversions)} failed")
    if successful_conversions > 0:
        success_rate = successful_conversions/(successful_conversions+len(failed_conversions))*100
        print(f"Conversion success rate: {success_rate:.1f}%")
    
    print(f"\nOutput structure:")
    print(f"├── {pose_files_dir}")
    print(f"├── {pickle_files_dir}")
    print(f"├── {mapping_file}")
    print(f"├── {class_mapping_file}")
    print(f"└── {summary_file}")
    
    if len(failed_conversions) > 0:
        print(f"\nFailed conversions (first 10): {failed_conversions[:10]}")
        print("Check if video filenames match video IDs in NSLT metadata")
    
    # Show successful gloss distribution
    if successful_conversions > 0:
        print(f"\nSuccessfully converted glosses:")
        gloss_counts = {}
        pickle_files = os.listdir(pickle_files_dir)
        for pkl_file in pickle_files[:20]:  # Show first 20
            if pkl_file.endswith('.pkl'):
                try:
                    with open(os.path.join(pickle_files_dir, pkl_file), 'rb') as f:
                        data = pickle.load(f)
                        gloss = data['gloss']
                        gloss_counts[gloss] = gloss_counts.get(gloss, 0) + 1
                        if len(gloss_counts) <= 10:  # Show first 10 unique glosses
                            print(f"  {data['video_id']} -> {data['gloss']} (action {data['primary_action']}, {data['split']})")
                except:
                    pass
        
        print(f"\nGloss distribution: {len(set(gloss_counts.keys()))} unique glosses")
        top_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for gloss, count in top_glosses:
            print(f"  {gloss}: {count} files")

def pose_to_numpy(pose):
    """Convert pose_format Pose object to numpy arrays"""
    try:
        if pose.body.data is None or len(pose.body.data) == 0:
            return None, None
        
        pose_data = pose.body.data
        
        if len(pose_data.shape) == 4:
            # Take first person if multiple people detected
            keypoints = pose_data[:, 0, :, :2]  # x, y coordinates
            confidences = pose_data[:, 0, :, 2] if pose_data.shape[-1] > 2 else np.ones(pose_data[:, 0, :, :2].shape[:2])
        else:
            keypoints = pose_data[:, :, :2] if pose_data.shape[-1] >= 2 else pose_data
            confidences = pose_data[:, :, 2] if pose_data.shape[-1] > 2 else np.ones(pose_data.shape[:2])
        
        return keypoints, confidences
        
    except Exception as e:
        print(f"Error converting pose data: {e}")
        return None, None

if __name__ == "__main__":
    main()