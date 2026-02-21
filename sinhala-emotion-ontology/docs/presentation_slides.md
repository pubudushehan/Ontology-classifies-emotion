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


In the context of a Turtle (
.ttl
) file, these are prefixes (namespaces) used to make the code readable. Instead of writing long Internet addresses (URIs) every time, we use short abbreviations like seo: or rdf:.

Here is the breakdown of the specific ones in your file:

1. rdf (Resource Description Framework)
Full URI: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
Meaning: This is the standard W3C language used to describe information on the web.
Common Use: You often see rdf:type (which is shortened to just a in Turtle files).
In your file: It defines the core grammar. For example, rdf:Property means "this is a relationship type."

2. rdfs (RDF Schema)
Full URI: <http://www.w3.org/2000/01/rdf-schema#>
Meaning: This provides the vocabulary to structure data—basically defining what "Classes" (categories) and "Properties" (attributes) exist.
In your file:
rdfs:Class: Defines a category (e.g., "Emotion" is a Class).
rdfs:label: A human-readable name for something (e.g., "ආඩම්බර").
rdfs:domain & rdfs:range: Rules for properties (e.g., seo:hasEmotion connects a Word to an Emotion).

3. seo (Sinhala Emotion Ontology)
Full URI: <http://www.semanticweb.org/sinhala-emotion-ontology#>
Meaning: This is a custom prefix created specifically for your project. It is not a global standard like rdf or rdfs.
In your file:
seo:Emotion: You are defining a new concept called "Emotion".
seo:Word: You are defining a new concept called "Word".
seo:hasEmotion: That specific relationship linking a Sinhala word to its emotion.

