import json
import os
from collections import Counter
from indicnlp.tokenize import indic_tokenize

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(base_dir, "data", "sinhala_samples.json")
lexicon_file = os.path.join(base_dir, "ontology", "lexicon.json")

# Parameters
TOP_N_CANDIDATES = 300  # Initial candidates to consider
FINAL_N = 100           # Final number of words per emotion
MIN_LENGTH = 3          # Increased min length to avoid tiny common particles

def extract_lexicon():
    print(f"Loading data from {data_file}...")
    if not os.path.exists(data_file):
        print("Data file not found!")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab_by_emotion = {
        "Happy": Counter(),
        "Sad": Counter(),
        "Angry": Counter(),
        # Neutral usually has too few samples or is defined by absence of others
        # We will largely ignore Neutral for lexicon generation unless we have distinct Neutral words
    }
    
    all_words_counter = Counter()

    print("Processing samples...")
    for item in data:
        text = item.get("text", "")
        emotion = item.get("expected", "")
        
        if emotion in vocab_by_emotion:
            tokens = indic_tokenize.trivial_tokenize(text)
            # Filter tokens
            valid_tokens = [t for t in tokens if len(t) >= MIN_LENGTH]
            
            vocab_by_emotion[emotion].update(valid_tokens)
            all_words_counter.update(valid_tokens)

    # 1. Identify common stopwords (globally frequent)
    # If a word is very frequent overall, it might be a stopword.
    # But some emotion words ARE frequent.
    # Better approach: Interaction check.
    
    # 2. Get candidates
    candidates = {}
    for emotion in vocab_by_emotion:
        # Get top N candidates
        candidates[emotion] = set([w for w, c in vocab_by_emotion[emotion].most_common(TOP_N_CANDIDATES)])
    
    # 3. Filter for Exclusivity
    # A word should only belong to ONE emotion in our Ontology.
    # If it appears in multiple candidate sets, remove from ALL.
    
    exclusive_candidates = {e: set(words) for e, words in candidates.items()}
    
    # Check overlaps
    emotions = list(candidates.keys())
    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            e1 = emotions[i]
            e2 = emotions[j]
            
            intersection = candidates[e1].intersection(candidates[e2])
            if intersection:
                print(f"Removing {len(intersection)} overlapping words between {e1} and {e2}")
                # Remove from exclusive sets
                exclusive_candidates[e1] -= intersection
                exclusive_candidates[e2] -= intersection
                
    # 4. Final Selection
    final_lexicon = {}
    for emotion in emotions:
        # Sort remaining exclusive candidates by frequency
        sorted_exclusive = sorted(list(exclusive_candidates[emotion]), 
                                  key=lambda w: vocab_by_emotion[emotion][w], 
                                  reverse=True)
        
        # Take top FINAL_N
        selected = sorted_exclusive[:FINAL_N]
        final_lexicon[emotion] = selected
        print(f"Selected {len(selected)} exclusive words for {emotion}")
        # print(f"Top 5: {selected[:5]}")

    # Save
    print(f"Saving exclusive lexicon to {lexicon_file}...")
    with open(lexicon_file, 'w', encoding='utf-8') as f:
        json.dump(final_lexicon, f, ensure_ascii=False, indent=4)
    
    print("Lexicon extraction complete.")

if __name__ == "__main__":
    extract_lexicon()
