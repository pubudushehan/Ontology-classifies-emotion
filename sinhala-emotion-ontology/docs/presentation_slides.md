# Research Evaluation Presentation Slides

## Slide 1: Objective 01 - Semantic Frame-Based Emotion Ontology

**Title:** 3-Tier Semantic Emotion Ontology for Sinhala

**Core Concept:**
A **frame-based** RDF/OWL ontology that classifies emotions by **meaning, not keywords** — using semantic roles, negation, intensifiers, and discourse context.

**Key Contributions:**
*   **18 Emotion Frames** (e.g., PhysicalHarm, PositiveEmotion, LossExperience) with role-dependent emotion mappings.
*   **Linguistic Modifiers:** Negation flipping, intensifier/diminisher scaling, contrastive discourse connectives.
*   **Semantic Role Detection:** Agent vs. Patient perspective changes the emotion (e.g., "I hit" → Angry, "I was hit" → Sad).

**3-Tier Pipeline:**

**Tier 1 – Linguistic Analysis:**
Before looking at emotions, the system scans the sentence for contextual clues — negation words (නෑ, නැහැ), intensifiers (හරිම, ගොඩක්), diminishers (පොඩ්ඩක්), discourse connectives like "but" (ඒත්, නමුත්), and semantic role markers that tell us *who* is the doer (Agent) and *who* is affected (Patient).

**Tier 2 – Frame-Based Ontology Matching:**
Each word is matched not to an emotion directly, but to an **Emotion Frame** (e.g., "ගහනවා" → PhysicalHarm, "සතුටුයි" → PositiveEmotion) via SPARQL queries on the RDF ontology. Each frame carries multiple emotion possibilities depending on the speaker's role, polarity, and what happens under negation.

**Tier 3 – Semantic Inference:**
The system combines everything: if the speaker is the Patient of a PhysicalHarm frame → Sad (not Angry). If a negation word is near a PositiveEmotion frame → flip to Sad. Intensifiers boost confidence, contrastive connectives like "but" make the latter clause dominate. The final emotion is the one with the strongest combined signal.

**Result:** 92.2% accuracy | Ontology handles 82.4% of decisions at 93.7% accuracy.

**Why it matters:**
Moves beyond keyword lookup to **context-aware, explainable** emotion reasoning — aligned with the Emotion Frame Ontology (De Giorgis et al., 2024).

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

