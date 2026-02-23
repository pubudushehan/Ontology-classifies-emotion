"""
evaluate.py -- Evaluate the 3-tier frame-based emotion classifier.

Reports:
  - Overall accuracy and per-class precision/recall/F1
  - Method distribution (ontology vs ML breakdown)
  - Feature impact: negation, intensifiers, discourse connectives, hostile address
  - Detailed per-frame statistics
"""

import json
import os
import random
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.classify import EmotionClassifier

# Set to None for full dataset, or a number for quick testing
SAMPLE_LIMIT = 500
RANDOM_SEED = 42


def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "sinhala_samples.json")

    print(f"Loading data from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter valid samples
    valid_data = [d for d in data if d.get("text") and d.get("expected")]
    count = len(valid_data)
    print(f"Total valid samples: {count}")

    if SAMPLE_LIMIT and count > SAMPLE_LIMIT:
        random.seed(RANDOM_SEED)
        print(f"Sampling {SAMPLE_LIMIT} random samples (seed={RANDOM_SEED})...")
        valid_data = random.sample(valid_data, SAMPLE_LIMIT)
    else:
        print("Using all samples.")

    # Initialize Classifier
    print("Initializing Classifier...")
    classifier = EmotionClassifier()

    y_true = []
    y_pred = []
    methods_used = Counter()
    method_correct = Counter()
    method_total = Counter()
    negation_cases = []
    intensifier_cases = []
    connective_cases = []
    hostile_cases = []
    frame_usage = Counter()

    print(f"Running classification on {len(valid_data)} samples...\n")
    for i, item in enumerate(valid_data):
        text = item["text"]
        expected = item["expected"]

        res = classifier.predict(text)
        predicted = res["label"]
        method = res.get("method", "")
        explanation = res.get("explanation", [])

        y_true.append(expected)
        y_pred.append(predicted)

        # Track method
        if "ML" in method:
            method_key = "ML"
        else:
            method_key = "Ontology"
        methods_used[method_key] += 1
        method_total[method_key] += 1
        if predicted == expected:
            method_correct[method_key] += 1

        # Track feature usage from explanation
        expl_text = " ".join(explanation)
        if "negated" in expl_text:
            is_correct = predicted == expected
            negation_cases.append((text, expected, predicted, is_correct))
        if "intensifier" in expl_text:
            is_correct = predicted == expected
            intensifier_cases.append((text, expected, predicted, is_correct))
        if "discourse" in expl_text:
            is_correct = predicted == expected
            connective_cases.append((text, expected, predicted, is_correct))
        if "Hostile" in expl_text:
            is_correct = predicted == expected
            hostile_cases.append((text, expected, predicted, is_correct))

        # Track frame usage
        for step in explanation:
            if "[" in step and "]" in step:
                frame_name = step.split("[")[1].split("]")[0]
                frame_usage[frame_name] += 1

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(valid_data)}...")

    # ===================================================================
    # Report
    # ===================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS: Frame-Based Emotion Ontology")
    print("=" * 60)

    # 1. Overall Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")

    # 2. Classification Report
    labels = sorted(set(y_true + y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"Confusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>10}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label:>10}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>10}", end="")
        print()

    # 4. Method Distribution
    print(f"\nMethod Distribution:")
    for method, count in methods_used.most_common():
        correct = method_correct[method]
        total = method_total[method]
        method_acc = correct / total if total > 0 else 0
        print(f"  {method}: {count} predictions ({count/len(y_true)*100:.1f}%), "
              f"Accuracy: {method_acc:.4f} ({correct}/{total})")

    # 5. Feature Impact
    print(f"\nSemantic Feature Impact:")

    if negation_cases:
        neg_correct = sum(1 for _, _, _, c in negation_cases if c)
        print(f"  Negation detected: {len(negation_cases)} cases, "
              f"Accuracy: {neg_correct/len(negation_cases):.4f} ({neg_correct}/{len(negation_cases)})")
        # Show some examples
        for text, exp, pred, correct in negation_cases[:3]:
            status = "OK" if correct else "WRONG"
            print(f"    [{status}] '{text[:50]}...' expected={exp}, predicted={pred}")
    else:
        print(f"  Negation: 0 cases detected")

    if intensifier_cases:
        int_correct = sum(1 for _, _, _, c in intensifier_cases if c)
        print(f"  Intensifiers: {len(intensifier_cases)} cases, "
              f"Accuracy: {int_correct/len(intensifier_cases):.4f} ({int_correct}/{len(intensifier_cases)})")
    else:
        print(f"  Intensifiers: 0 cases detected")

    if connective_cases:
        conn_correct = sum(1 for _, _, _, c in connective_cases if c)
        print(f"  Discourse connectives: {len(connective_cases)} cases, "
              f"Accuracy: {conn_correct/len(connective_cases):.4f} ({conn_correct}/{len(connective_cases)})")
    else:
        print(f"  Discourse connectives: 0 cases detected")

    if hostile_cases:
        host_correct = sum(1 for _, _, _, c in hostile_cases if c)
        print(f"  Hostile address: {len(hostile_cases)} cases, "
              f"Accuracy: {host_correct/len(hostile_cases):.4f} ({host_correct}/{len(hostile_cases)})")
    else:
        print(f"  Hostile address: 0 cases detected")

    # 6. Frame Usage
    print(f"\nTop 15 Most Used Frames:")
    for frame_name, count in frame_usage.most_common(15):
        print(f"  {frame_name}: {count}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
