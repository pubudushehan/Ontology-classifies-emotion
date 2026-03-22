# Sinhala Emotion Ontology Project 🇱🇰

A hybrid emotion classification system for Sinhala text, combining an advanced **3-Tier Semantic Frame-based Ontology** (RDF/OWL) and **Machine Learning embeddings** (LaBSE). This project allows users to classify Sinhala sentences into four emotions: **Happy** (සතුට), **Sad** (දුක), **Angry** (කෝප), and **Neutral** (සාමාන්‍ය).

## 🚀 Features
- **Hybrid Classification**:
  - **Ontology (3-Tier Semantic Inference)**: Uses a structured knowledge base (RDF) built around `EmotionFrame`s mapping tokens to emotions based on precise matching. Considers Linguistic features like negations, intensifiers, diminishers, discourse connectives, and semantic roles (agent/patient/experiencer).
  - **Machine Learning (LaBSE)**: Uses Language-agnostic BERT Sentence Embeddings to classify text based on semantic similarity to known emotion centroids (acting as a fallback for conflicting frames or unseen words).
- **FastAPI Backend**: A high-performance web API to serve the classification model.
- **Interactive UI**: Swagger UI for easy testing and demonstrations.
- **Data-Driven**: Trained on ~3,500 real Sinhala voice cut samples.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | **Python** (3.9+) | Core programming language. |
| **Ontology** | **RDFLib** | Creating and querying the RDF/OWL ontology graph using SPARQL. |
| **Embeddings** | **Sentence-Transformers** | Loading the `LaBSE` model for semantic text vectorization. |
| **Web API** | **FastAPI** | Exposing the classification logic as a REST endpoint. |
| **Server** | **Uvicorn** | ASGI server to run the FastAPI application. |
| **NLP** | **IndicNLP Library** | Tokenizing Sinhala text efficiently. |
| **Math** | **NumPy** | Vector operations for cosine similarity calculations. |

---

## 📂 Project Structure & File Descriptions

Here is a detailed breakdown of the project files:

### 1. Source Code (`src/`)
*   **`create_ontology.py`**:
    *   **Purpose**: Generates the Frame-based Sinhala Emotion Ontology (RDF/Turtle).
    *   **Details**: Reads `frames.json` and `modifiers.json` and uses `RDFLib` to define classes (`EmotionFrame`, `LexicalTrigger`, `NegationMarker`, etc.) and properties (`hasTypicalEmotion`, `hasAgentEmotion`, etc.).
*   **`build_model.py`**:
    *   **Purpose**: Pre-calculates emotion centroids for the Machine Learning classifier.
    *   **Details**: Loads the training data, encodes all sentences using LaBSE, calculates the average vector (centroid) for each emotion (Happy, Sad, Angry), and saves them to `data/centroids.pkl`.
*   **`classify.py`**:
    *   **Purpose**: The core 3-Tier semantic hybrid classifier.
    *   **Details**:
        1. **Tier 1 (Linguistic Analysis)**: Detects negation, intensifiers, diminishers, connectives, and roles via case markers.
        2. **Tier 2 (Frame-Based Ontology Matching)**: Matches tokens to `EmotionFrame`s via SPARQL query.
        3. **Tier 3 (Semantic Inference)**: Combines matched frames with linguistic markers to accumulate weighted confidence scores for each emotion.
        4. **Fallback**: If there is no clear dominating ontology match (or no matches at all), it defaults to LaBSE cosine similarity against the pre-computed ML centroids.
*   **`app.py`**:
    *   **Purpose**: The Web Application entry point.
    *   **Details**: Initializes the `EmotionClassifier` and defines the `/classify` API endpoint. Serves Swagger UI at `/docs`.
*   **`evaluate.py`**:
    *   **Purpose**: Evaluation script.
    *   **Details**: Runs the classifier against a random sample of the dataset to calculate Accuracy and F1-scores.

### 2. Data & Resources (`data/` & `ontology/`)
*   **`ontology/sinhala_emotion.ttl`**:
    *   The generated Knowledge Graph in Turtle format. Contains the relationships and rules.
*   **`ontology/frames.json`**:
    *   Defines semantic frames containing triggers, corresponding agent/patient emotions, and polarity configuration.
*   **`ontology/modifiers.json`**:
    *   Defines syntactic modifiers like intensifiers, diminishers, discourse connectives, and negations, alongside their effect weights.
*   **`ontology/role_markers.json`**:
    *   Lists grammatical markers to decide active vs. passive sentence role (agent/patient/experiencer) and hostile address types.
*   **`data/sinhala_samples.json`**:
    *   The main dataset containing ~3,500 labeled examples (merged from Voice Cuts).
*   **`data/centroids.pkl`**:
    *   A serialized binary file containing the computed vector centroids for 'Happy', 'Sad', and 'Angry'.

### 3. Root Files
*   **`requirements.txt`**: List of all Python dependencies required to run the project.
*   **`README.md`**: Project documentation (this file).

---

## 🏃‍♂️ Setup & Installation

1.  **Clone/Navigate to Project**:
    ```bash
    cd sinhala-emotion-ontology
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate Resources** (Run these once):
    ```bash
    # 1. Generate Semantic Frame-Based Ontology Graph
    python src/create_ontology.py

    # 2. Build ML Model (Centroids)
    python src/build_model.py
    ```

---

## 🎮 Usage

### Start the API Server
Run the following command to start the web application:
```bash
uvicorn src.app:app --reload
```

### Access the Interface
Open your browser and navigate to:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

You will see the **Swagger UI** where you can examine the API and execute tests.

### API Endpoint: `/classify`
- **Method**: `GET`
- **Parameter**: `text` (The Sinhala sentence)
- **Example Response**:
    ```json
    {
        "label": "Sad",
        "confidence": 0.8,
        "method": "Ontology (Frame-based, 2 triggers)",
        "matched_words": {
            "Sad": ["දුකයි"]
        },
        "explanation": [
            "'දුකයි' [frame_sadness] negated -> Sad",
            "  intensifier boost x1.5"
        ]
    }
    ```

---

## 📊 Evaluation
To check the accuracy of the model on the dataset:
```bash
python src/evaluate.py
```

---

## 📚 References
1.  **InfoLabs (Voice Cuts)**: Source of the raw Sinhala audio/transcript data.
2.  **Sentence-Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
3.  **FastAPI**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
4.  **RDFLib**: [https://rdflib.readthedocs.io/](https://rdflib.readthedocs.io/)
5.  **IndicNLP**: [https://anoopkunchukuttan.github.io/indic_nlp_library/](https://anoopkunchukuttan.github.io/indic_nlp_library/)
6.  **Emotion Frame Ontology (EFO)**: De Giorgis & Gangemi 2024
7.  **TONE 3-Tiered Ontology for Emotion**: 2024

---

### 🔍 The "Deep Flow" of Classification
Here is exactly what happens inside the system when you input: `"මම ගොඩක් සතුටු නෑ"` (I'm not very happy)

**1. Tokenization (Breaking it down)**
The system first uses IndicNLP to split your sentence into individual words (tokens): 
`['මම', 'ගොඩක්', 'සතුටු', 'නෑ']`

**2. Tier 1: Linguistic Analysis**
The system identifies structural modifiers present in the sentence (using `modifiers.json` and `role_markers.json`).
*   `'මම'` identifies the speaker as an 'agent'.
*   `'ගොඩක්'` is mapped to an **intensifier** with a predefined multiplier token.
*   `'නෑ'` triggers a **negation** flag within a specified window buffer.

**3. Tier 2: Ontology Frame Matching (SPARQL)**
The system ignores words acting as modifiers and triggers SPARQL queries on the remaining tokens to discover matching internal `EmotionFrame`s.
*   `'සතුටු'` triggers a match on the `frame_happiness` schema (with default 'Happy' base emotion and specific mapping to 'Sad' upon 'negatedEmotion').

**4. Tier 3: Semantic Inference**
The system combines frames from Tier 2 with structural modifiers from Tier 1:
*   Instead of 'Happy', the `'සතුටු'` frame gets flipped to 'Sad' because of the proximity of the negator (`'නෑ'`).
*   The raw weight of the frame is multiplied because the intensifier `'ගොඩක්'` boosts the sentiment score calculation.
*   All valid predictions are accumulated across the sentence.

**5. Decision Rules Output**
*   Because Ontology-driven inference resolves clearly pointing to **Sad**, the engine returns `Sad` as the emotion and includes an explanation trail showing the multiplier boosts and negation flips. 
*   *(If multiple conflicting emotions matched with comparable weights, the engine would have skipped the ontology scores entirely and defaulted to executing an ML fallback by generating a LaBSE vector for the sentence and comparing it via Cosine Similarity against the ML Centroids computed in `data/centroids.pkl`)*