import json
import os

# Paths
base_dir = "/Users/pubudushehan/Downloads/reseach_LR/Research_Voice_Cut"
voice_cuts_dir = os.path.join(base_dir, "Voice Cuts")
target_file = os.path.join(base_dir, "sinhala-emotion-ontology/data/sinhala_samples.json")

# Mapping filenames to labels
files_map = {
    "Angry Voice Cut.json": "Angry",
    "Happy Voice Cut.json": "Happy",
    "Sad Voice Cut.json": "Sad"
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    # Load existing samples
    if os.path.exists(target_file):
        print(f"Loading existing data from {target_file}")
        all_samples = load_json(target_file)
    else:
        all_samples = []

    print(f"Initial sample count: {len(all_samples)}")

    # Load new data
    for filename, label in files_map.items():
        file_path = os.path.join(voice_cuts_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        try:
            data = load_json(file_path)
            # Check structure. We saw {"dataset": [{"text": "...", ...}]}
            if "dataset" in data:
                items = data["dataset"]
                print(f"Loaded {len(items)} items from {filename}")
                
                count = 0
                for item in items:
                    if "text" in item:
                        # cleanup text
                        text = item["text"].strip()
                        if text:
                            # Avoid duplicates?
                            # For now, just append. Maybe check if text already exists could be slow if simple list.
                            # Let's just create a set of existing texts first.
                            
                            all_samples.append({
                                "text": text,
                                "expected": label
                            })
                            count += 1
                print(f"Added {count} samples for {label}")
            else:
                print(f"Warning: Unexpected structure in {filename}. Keys: {data.keys()}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Deduplicate based on text
    unique_samples = {}
    for s in all_samples:
        unique_samples[s['text']] = s

    final_list = list(unique_samples.values())
    
    print(f"Final distinct sample count: {len(final_list)}")
    
    # Save
    save_json(target_file, final_list)
    print("Data import complete.")

if __name__ == "__main__":
    main()
