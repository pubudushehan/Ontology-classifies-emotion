# Research Justification: Hybrid Ontology-Transformer Approach

OWL stands for Web Ontology Language.

This document justifies the architectural decisions of the **Sinhala Emotion Ontology Project** using evidence from recent academic research. We have adopted a **Hybrid Approach** combining an **OWL-based Ontology** (for rule-based precision) with **Transformer-based Deep Learning** (LaBSE for contextual ambiguity).

## 1. Why an Ontology? (The Rule Layer)

**Argument:** Emotions are complex socio-cognitive frames, not just statistical labels. An ontology allows us to formalize these concepts (Happy, Sad, Angry) and their linguistic triggers (Keywords) into a structured knowledge base that enables reasoning.

**Evidence from Research:**
> *"We propose an OWL frame-based ontology of emotions: the Emotion Frames Ontology (EFO)... EFO follows pattern-based ontology design... and is used to model multiple emotion theories."*
> — **De Giorgis & Gangemi (2024)**, *The Emotion Frame Ontology*, [arXiv:2401.10751]

**Relevance into Our Project:**
*   Following De Giorgis et al., we implemented a **Lexicon-based Ontology** (`lexicon.json` -> `sinhala_emotion.ttl`) to serve as the "First Line of Defense".
*   As noted in *Eisenreich et al. (2012)*, an ontology-based approach allows for **"Incremental Annotation"**, where we can map specific narrative elements (words) to emotional states. This is implemented in our `classify_ontology` function, which maps specific words to emotional classes effectively.

## 2. Why Deep Learning / Transformers? (The Context Layer)

**Argument:** Ontologies and keyword lists fail when sentences contain conflicting emotions, sarcasm, or implicit meaning. Deep Learning models, specifically those inspired by biological processes, are required to capture these hidden features.

**Evidence from Research:**
> *"Deep learning discusses about the supervised and unsupervised machine learning techniques in which they learn automatically to interpret the hierarchical representations... Deep learning has the consideration of the research community team due to the inspiration from the biological observations on human brain processing system."*
> — **Anushya & Nisha Priya (2020)**, *Mood recognition emotion ontology with texts*, [IJARIIT]

**Relevance into Our Project:**
*   Our system uses **LaBSE (Language-Agnostic BERT Sentence Embedding)**, a Transformer model.
*   As Anushya et al. highlight, traditional keyword methods are "tough to examine the fundamental behavior" of complex social media text.
*   By using LaBSE, we handle the **"Conflict Resolution"** phase: when the Ontology finds mixed signals (e.g., both "Happy" and "Sad" keywords), the Deep Learning model resolves the ambiguity.

## 3. The Power of the Hybrid Model

**Argument:** Neither approach is sufficient alone for a low-resource language like Sinhala.
*   **Pure ML** requires massive labeled datasets (which we lack).
*   **Pure Ontology** is too rigid and misses context.
*   **Hybrid** allows "Semantic Injection" (Ontology) into "Statistical Inference" (ML).

**Proof of Concept:**
Our implementation directly reflects the workflow described by **Eisenreich et al. (2012)** in *From Tale to Speech*:
> *"Our system first parses the input... extracts as much relevant information as possible on the characters – including their emotions... This provides us with an annotated version... used for populating the ontology."*

Similarly, our `classify.py`:
1.  **Extracts**: Tokenizes and checks Ontology (Eisenreich's "Annotation").
2.  **Populates**: If found, assigns label (De Giorgis's "Frame Activation").
3.  **Refines**: If ambiguous, uses Vector Embeddings (Anushya's "Deep Learning").

## 4. References

1.  **De Giorgis, S., & Gangemi, A.** (2024). *The Emotion Frame Ontology*. arXiv preprint arXiv:2401.10751.
2.  **Eisenreich, C., Ott, J., Süßdorf, T., Willms, C., & Declerck, T.** (2012). *From Tale to Speech: Ontology-based Emotion and Dialogue Annotation of Fairy Tales with a TTS Output*. DFKI & Saarland University.
3.  **Anushya, P. J., & Nisha Priya, P.** (2020). *Mood recognition emotion ontology with texts*. International Journal of Advance Research, Ideas and Innovations in Technology (IJARIIT), 6(4).
