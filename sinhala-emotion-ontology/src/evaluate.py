import json
import os
import random
from sklearn.metrics import classification_report, accuracy_score
from src.classify import EmotionClassifier

# Limit samples for quick check (set to None for full evaluation)
SAMPLE_LIMIT = 50 

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "sinhala_samples.json")
    
    print(f"Loading data from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter valid samples
    valid_data = [d for d in data if d.get("text") and d.get("expected")]
    count = len(valid_data)
    print(f"Total samples: {count}")
    
    if SAMPLE_LIMIT and count > SAMPLE_LIMIT:
        print(f"Sampling {SAMPLE_LIMIT} random samples for evaluation...")
        valid_data = random.sample(valid_data, SAMPLE_LIMIT)
    else:
        print("Using all samples.")

    # Initialize Classifier
    print("Initializing Classifier...")
    classifier = EmotionClassifier()
    
    y_true = []
    y_pred = []
    
    print("Running classification...")
    for item in valid_data:
        text = item["text"]
        expected = item["expected"]
        
        # Predict
        res = classifier.predict(text)
        predicted = res["label"]
        
        y_true.append(expected)
        y_pred.append(predicted)
        
        # Print sample
        print(f"Text: {text[:30]}... | Expected: {expected} | Predicted: {predicted} ({res['confidence']})")

    # Metrics
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    evaluate()
