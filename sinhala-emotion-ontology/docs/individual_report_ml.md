### 3.6 Machine Learning Model Creation

**Contribution Level:** Full Participation with Code Repository Contributions

The ML model development phase involved multiple components working in an integrated pipeline. Like the ontology creation, this was implemented collaboratively through GitHub repository commits.

### 3.6.1 Hybrid Emotion Detection Architecture

**LaBSE-Based Classification Model:**
I participated in developing the deep learning component of emotion detection:

**Model Architecture:**
*   **Base Model:** Language-Agnostic BERT Sentence Embedding (LaBSE)
*   **Fine-tuning Strategy:** Unsupervised centroid calculation using labeled Sinhala text (Fine-tuning was done by calculating the mean embedding core for each emotion cluster, rather than supervised gradient updates)
*   **Classification Head:** Centroid-based classification using Cosine Similarity for disambiguation
*   **Hybrid Fallback:** Fallback integration with frame-based ontology for ambiguous cases

**My Contributions:**
*   **Preparing training data:** Organizing voice-cut transcriptions and augmenting the dataset using the ontology's `lexicon.json` for enhanced centroid calculation
*   **Implementing centroid-based classification:** Developing the logic to compute and serialize (`centroids.pkl`) the mean normalized vector for each emotion
*   **Developing the hybrid fallback mechanism:** Integrating ML classification when ontology rules return no match or conflicting matches
*   **Testing and evaluating classification accuracy:** Comparing the predictions against the expected labels
*   **Committing model training scripts:** Adding scripts (`build_model.py`, `classify.py`) and configurations to the GitHub repository

**Processing Pipeline:**

| Stage | Process Description |
| :--- | :--- |
| **Input Layer** | Receive plain Sinhala text input |
| **Embedding Layer** | `LaBSE` generates a contextualized sentence embedding |
| **Classification Layer** | Centroid-based classification computes Cosine Similarity against serialized emotion centroids |
| **Ontology Check** | Use rule-based triggers and frames. If confidence is 0 or multiple emotions trigger equally, Fallback to ML Classification. |
| **Output** | Emotion label + confidence score |



