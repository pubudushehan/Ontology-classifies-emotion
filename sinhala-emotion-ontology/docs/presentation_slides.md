# Research Evaluation Presentation Slides

## Slide 1: Objective 01 - Ontology Creation for Emotion Detection

**Title:** Development of the Sinhala Emotion Ontology (SEO)

**Core Concept:**
A formal, rule-based framework to represent emotional concepts and their linguistic triggers in Sinhala, built using **OWL (Web Ontology Language)**.

**Key Contributions:**
*   **Structured Knowledge Base:** Modeled 4 core emotion classes (Happy, Sad, Angry, Neutral) as a hierarchical ontology.
*   **Lexicon Integration:** Mapped **[Number]** Sinhala keywords (e.g., "සතුට" -> Happy) directly to ontology classes.
*   **Deterministic Reasoning:** specifically designed to handle **explicit emotional triggers** with 100% precision.
*   **Standard Compliance:** Aligned with the **Information Artifact Ontology (IAO)** and **Emotion Frame Ontology (EFO)** principles (De Giorgis et al., 2024).

**Methodology:**
1.  **Extraction:** Collected domain-specific keywords from the "Voice Cut" dataset.
2.  **Formalization:** Converted keywords into RDF/OWL triples (`Word` -> `hasEmotion` -> `Emotion`).
3.  **Visualization:** Graph-based representation of emotional relationships.

**Why this matters:**
Provides a transparent, explainable "First Line of Defense" for classification, ensuring that known emotional words are never misclassified by the "black box" of Deep Learning.

---

## Slide 2: Objective 02 - Fine-Tuning Transformers for Emotion Detection

**Title:** Deep Learning Approach: Context-Aware Classification

**Core Concept:**
Utilization of **Transformer-based Language Models** to capture context, ambiguity, and implicit emotions that rule-based ontologies miss.

**Model Used:**
*   **LaBSE (Language-Agnostic BERT Sentence Embedding)**
*   *Note: Chosen for its superior performance on low-resource languages like Sinhala compared to standard BERT.*

**Key Contributions:**
*   **Centroid-Based Classification:** computed "Emotion Centroids" (Average Vectors) from **3,500+** Sinhala audio transcripts.
*   **Contextual Disambiguation:** Handles sentences with conflicting keywords (e.g., "I am happy about the result but sad about the cost") by analyzing the full sentence vector.
*   **Hybrid Fallback:** Acts as the resolution engine when the Ontology fails or detects conflicts.

**Methodology:**
1.  **Embedding:** Converted input sentences into 768-dimensional vectors.
2.  **Training/Clustering:** Grouped vectors by emotion to find the mathematical "center" of "Happy", "Sad", and "Angry" in Sinhala.
3.  **Inference:** Calculated **Cosine Similarity** between new inputs and these centroids to determine the closest emotion.

**Why this matters:**
Achieves robustness in real-world scenarios where language is subtle, sarcastic, or mixed, complementing the rigidity of the Ontology (Anushya et al., 2020).
