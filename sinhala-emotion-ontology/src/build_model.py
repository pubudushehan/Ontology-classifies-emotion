import json
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(base_dir, "data", "sinhala_samples.json")
output_file = os.path.join(base_dir, "data", "centroids.pkl")

# Model
MODEL_NAME = "sentence-transformers/LaBSE"

def build_model():
    print(f"Loading data from {data_file}...")
    if not os.path.exists(data_file):
        print("Data file not found!")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Organize data by label
    # We only care about Happy, Sad, Angry for now as per data availability.
    # If Neutral is present, we include it.
    
    samples_by_label = {}
    for item in data:
        label = item.get("expected")
        text = item.get("text")
        if label and text:
            if label not in samples_by_label:
                samples_by_label[label] = []
            samples_by_label[label].append(text)

    print(f"Found labels: {list(samples_by_label.keys())}")
    for label, texts in samples_by_label.items():
        print(f"  {label}: {len(texts)} samples")

    # Load Model
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    centroids = {}
    
    for label, texts in samples_by_label.items():
        print(f"Computing embeddings for {label}...")
        embeddings = model.encode(texts)
        # Compute mean
        centroid = np.mean(embeddings, axis=0)
        # Normalize centroid? Cosine similarity works on direction, but usually good to keep raw or normalized.
        # Let's normalize for easier cosine similarity calc (dot product).
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid

    # Save
    print(f"Saving centroids to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(centroids, f)
    
    print("Model build complete.")

if __name__ == "__main__":
    build_model()
