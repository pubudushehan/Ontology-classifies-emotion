### 3.5.3 Frame-Based Ontology Development

**Proposed Solution:**
Based on my literature review findings, I proposed creating a frame-based ontology following the EFO methodology adapted for Sinhala emotional TTS. This approach represented a significant methodological advancement.

**Sinhala Emotion Ontology (SEO) Framework:**
I designed and implemented the following frame-based structure:

*   **Emotion Frames:** Core emotional concepts (Happy, Sad, Angry, Neutral) modeled as semantic frames
*   **Semantic Roles:**
    *   *Agent:* The doer of the action (Nominative case)
    *   *Patient:* The receiver of an action (often the stimulus or target of the emotion)
    *   *Experiencer:* The perspective from which emotion is felt (to whom something happens)
    *   *Possessive:* Indicating ownership or relationship
*   **Frame Elements:** Specific situations that instantiate emotion frames
    *   *PhysicalHarm* frame → triggers Anger/Sadness
    *   *PleasantExperience* frame → triggers Happiness
    *   *LossExperience* frame → triggers Sadness
    *   *VerbalInsult / ThreatAction* frames → triggers Anger
*   **Linguistic Modifiers:**
    *   *Negation handling* (flipping emotion polarity)
    *   *Intensifiers/Diminishers* (scaling emotional intensity)
    *   *Contrastive discourse connectives* (complex emotional transitions)

**Three-Tier Processing Pipeline:**
I designed the ontology to operate through a three-stage inference process:
1.  **Linguistic Analysis:** Parse Sinhala text to identify syntactic structures, semantic roles, and linguistic features
2.  **Frame Matching:** Map identified linguistic patterns to relevant emotion frames based on semantic similarity
3.  **Semantic Inference:** Apply inference rules to determine final emotion label considering context, negation, and intensity

**Implementation and GitHub Commitment:**
After designing the frame-based ontology architecture:
*   Implemented the ontology structure in Python using semantic web libraries
*   Created frame definitions and semantic role mappings
*   Developed inference rules for emotion classification
*   Tested the ontology on sample Sinhala sentences
*   Committed the complete implementation to our GitHub repository

The commit history in our GitHub repository shows:
1.  Initial keyword-based ontology commit (team effort)
2.  My literature review notes and proposed frame-based design
3.  Frame-based ontology implementation commit
4.  Iterative refinements based on testing (collaborative)

**Accuracy Improvement:**
The frame-based ontology demonstrated significantly improved performance compared to the keyword approach:
*   Better handling of contextual emotion (reduced false positives)
*   Proper negation and contrastive structure processing
*   Modeling of emotional intensity variations
*   Culturally appropriate Sinhala emotion patterns captured
