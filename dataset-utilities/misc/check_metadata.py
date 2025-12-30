import json
import os

# Check the structure of your new WLASL metadata files
metadata_dir = r"C:/Users/padwe/OneDrive/WLASL-proj/wlasl-kaggle"

files_to_check = ['nslt_100.json', 'nslt_2000.json', 'WLASL_v0.3.json']

for filename in files_to_check:
    filepath = os.path.join(metadata_dir, filename)
    
    if os.path.exists(filepath):
        print(f"\n=== {filename} ===")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print(f"Data type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Top-level keys: {list(data.keys())}")
                
                # Check first entry to understand structure
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                print(f"First entry key: {first_key}")
                print(f"First entry value type: {type(first_value)}")
                
                if isinstance(first_value, dict):
                    print(f"First entry keys: {list(first_value.keys())}")
                elif isinstance(first_value, list) and len(first_value) > 0:
                    print(f"First list item: {first_value[0]}")
                    if isinstance(first_value[0], dict):
                        print(f"List item keys: {list(first_value[0].keys())}")
                
                print(f"Total number of entries: {len(data)}")
                
            elif isinstance(data, list):
                print(f"List length: {len(data)}")
                if len(data) > 0:
                    print(f"First item: {data[0]}")
                    if isinstance(data[0], dict):
                        print(f"Item keys: {list(data[0].keys())}")
                        
                        # Check if it has the expected structure
                        if 'gloss' in data[0]:
                            print(f"First gloss: {data[0]['gloss']}")
                        if 'instances' in data[0]:
                            print(f"Number of instances for first gloss: {len(data[0]['instances'])}")
                            if len(data[0]['instances']) > 0:
                                instance = data[0]['instances'][0]
                                print(f"First instance keys: {list(instance.keys())}")
                                if 'video_id' in instance:
                                    print(f"First video_id: {instance['video_id']}")
                        
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    else:
        print(f"\n{filename} not found at {filepath}")

# Also check the class list file
class_list_file = os.path.join(metadata_dir, 'wlasl_class_list.txt')
if os.path.exists(class_list_file):
    print(f"\n=== wlasl_class_list.txt ===")
    with open(class_list_file, 'r') as f:
        lines = f.readlines()
    print(f"Number of classes: {len(lines)}")
    print(f"First 10 classes: {[line.strip() for line in lines[:10]]}")
