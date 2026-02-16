# Sinhala Emotion Ontology Project 🇱🇰

A hybrid emotion classification system for Sinhala text, combining **Ontology-based rules** (RDF/OWL) and **Machine Learning embeddings** (LaBSE). This project allows users to classify Sinhala sentences into four emotions: **Happy** (සතුට), **Sad** (දුක), **Angry** (කෝප), and **Neutral** (සාමාන්‍ය).

## 🚀 Features
- **Hybrid Classification**:
  - **Ontology (Rule-based)**: Uses a structured knowledge base (RDF) to identify emotions based on explicit trigger words (100% confidence).
  - **Machine Learning (LaBSE)**: Uses Language-agnostic BERT Sentence Embeddings to classify text based on semantic similarity to known emotion centroids (for unseen words).
- **FastAPI Backend**: A high-performance web API to serve the classification model.
- **Interactive UI**: Swagger UI for easy testing and demonstrations.
- **Data-Driven**: Trained on ~3,500 real Sinhala voice cut samples.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | **Python** (3.9+) | Core programming language. |
| **Ontology** | **RDFLib** | Creating and querying the RDF/OWL ontology graph. |
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
    *   **Purpose**: Generates the RDF/OWL Ontology file (`sinhala_emotion.ttl`).
    *   **Details**: Reads `lexicon.json` and uses `RDFLib` to define classes (`Emotion`, `Word`) and properties (`hasEmotion`), linking Sinhala words to their respective emotions.
*   **`build_model.py`**:
    *   **Purpose**: Pre-calculates emotion centroids for the Machine Learning classifier.
    *   **Details**: Loads the training data, encodes all sentences using LaBSE, calculates the average vector (centroid) for each emotion (Happy, Sad, Angry), and saves them to `data/centroids.pkl`.
*   **`classify.py`**:
    *   **Purpose**: The core hybrid classifier.
    *   **Details**:
        1.  First, checks the Ontology for exact keyword matches.
        2.  If no match, encodes the input text and calculates cosine similarity against the pre-computed centroids.
        3.  Returns the label with the highest similarity score.
*   **`app.py`**:
    *   **Purpose**: The Web Application entry point.
    *   **Details**: Initializes the `EmotionClassifier` and defines the `/classify` API endpoint. Serves Swagger UI at `/docs`.
*   **`evaluate.py`**:
    *   **Purpose**: Evaluation script.
    *   **Details**: Runs the classifier against a random sample of the dataset to calculate Accuracy and F1-scores.

### 2. Data & Resources (`data/` & `ontology/`)
*   **`ontology/sinhala_emotion.ttl`**:
    *   The generated Knowledge Graph in Turtle format. Contains the relationships between words and emotions.
*   **`ontology/lexicon.json`**:
    *   A manual dictionary of key Sinhala emotion words used to build the ontology.
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
    # 1. Generate Ontology Graph
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
      "text": "මම ගොඩක් සතුටින්",
      "emotion": "Happy",
      "confidence": 1.0,
      "method": "Ontology"
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
