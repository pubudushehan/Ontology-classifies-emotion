# Sinhala Emotion Classification: Logic Flow Explained

This document explains exactly how the system decides whether a sentence is **Happy**, **Sad**, **Angry**, or **Neutral**. It uses a **Hybrid Approach**, meaning it combines strict Rules (Ontology) with AI intuition (Machine Learning).

---

## 🌊 High-Level Flow

1.  **Input**: Receive Sinhala Sentence.
2.  **Tokenize**: Break sentence into words.
3.  **Check Ontology**: Scan *every* specific word against our Knowledge Base.
4.  **Analyze Matches**:
    *   **Case A (Clear Match)**: All matching words point to *one* emotion. -> **Return that Emotion**.
    *   **Case B (Conflict)**: Words point to *different* emotions (e.g., "Sad" word + "Happy" word). -> **Go to ML**.
    *   **Case C (No Match)**: No words found in Ontology. -> **Go to ML**.
5.  **ML (Machine Learning)**: If needed, calculate the sentence's meaning (vector) and compare it to emotion averages.

---

## 🛠️ Step-by-Step with Code

### Step 1: Input & Tokenization
The system takes the raw text and uses `IndicNLP` to split it into specific tokens (words/punctuation).

**Code (`src/classify.py`):**
```python
from indicnlp.tokenize import indic_tokenize

def tokenize(self, text):
    # Input: "අද හවසට ක්රිකට් ගහන්න සෙට් වෙමු"
    # Output: ['අද', 'හවසට', 'ක්රිකට්', 'ගහන්න', 'සෙට්', 'වෙමු']
    return indic_tokenize.trivial_tokenize(text)
```

### Step 2: The Ontology Scan (Rule Layer)
We loop through **every token** and query the RDF Ontology (`sinhala_emotion.ttl`) to see if it's a known emotion trigger.

**Code (`src/classify.py`):**
```python
emotion_counts = {}

for token in tokens:
    # SPARQL query checks if this token is a defined 'Word' with an 'hasEmotion' link
    query = """
    SELECT ?emotion_label
    WHERE {
        ?w rdf:type seo:Word .
        ?w rdfs:label ?label .
        ?w seo:hasEmotion ?e .
        ?e rdfs:label ?emotion_label .
        FILTER(str(?label) = ?target)
    }
    """
    # ... execute query ...
    
    # If "සෙට්" is found, it returns "Happy". We count it.
    if emotion in emotion_counts:
        emotion_counts[emotion] += 1
```

### Step 3: Decision Logic (The Brain)
Here is where we decide if we trust the Ontology or need the AI.

**Code (`src/classify.py`):**
```python
# checking the counts we collected in Step 2 through 3 scenarios:

# SCENARIO 1: No Ontology matches found
if not emotion_counts:
    return self.classify_ml(text) 

found_emotions = list(emotion_counts.keys())

# SCENARIO 2: Single clear emotion (e.g., {'Happy': 2})
if len(found_emotions) == 1:
    return found_emotions[0] # "Happy"

# SCENARIO 3: Conflict (e.g., {'Happy': 1, 'Sad': 1})
else:
    # Data is confusing (mixed signals). Trust the ML model for context.
    return self.classify_ml(text)
```

### Step 4: Machine Learning Fallback (LaBSE)
If we reach this step (due to No Match or Conflict), we use the **LaBSE** model.
1.  Convert sentence to a **Vector** (list of numbers representing meaning).
2.  Compare this vector to the **Centroids** (Average vectors) of Happy, Sad, and Angry we built from the training data.
3.  Calculate **Dot Product** (Similarity Score).

**Code (`src/classify.py`):**
```python
def classify_ml(self, text):
    # 1. Provide Vector
    embedding = self.model.encode(text) 

    best_label = "Neutral"
    best_score = -1.0
    
    # 2. Compare with Happy/Sad/Angry Centroids
    for label, centroid in self.centroids.items():
        score = np.dot(embedding, centroid) # Similarity calculation
        
        if score > best_score:
            best_score = score
            best_label = label
            
    # 3. Handle Low Confidence (Neutral)
    if best_score < 0.25:
        return "Neutral"
        
    return best_label
```

---

## 📊 Example Trace

**Input**: `"මට හරිම සතුටුයි ඒත් දුකයි"` (*I am very happy but sad*)

1.  **Tokenize**: `['මට', 'හරිම', 'සතුටුයි', 'ඒත්', 'දුකයි']`
2.  **Ontology Scan**:
    *   `සතුටුයි` -> Matches **Happy**
    *   `දුකයි` -> Matches **Sad**
    *   `emotion_counts` = `{'Happy': 1, 'Sad': 1}`
3.  **Decision**:
    *   Is it empty? No.
    *   Is it single? No (`len` is 2).
    *   **Action**: Call `classify_ml()` (Conflict Resolution).
4.  **ML**:
    *   LaBSE reads the *whole* sentence meaning.
    *   It sees "Happy but Sad" and decides which feeling is stronger based on training patterns.
    *   Returns the dominant emotion (e.g., **Sad**).
