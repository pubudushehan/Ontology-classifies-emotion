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


Below is a copy-paste-ready report. Replace the bracketed details with your own information.

INDIVIDUAL REPORT

Semantic-Aware Emotional Text-to-Speech for Sinhala: A Single-Speaker Deep Learning Approach

Prepared by: [Your Name]
Index Number: [Your Index Number]
Group Number: 26

Group Members
S.P.A.S Jayasiri – ICT/21/860
A.V.N.J Pemarathna – ICT/21/896
K.A.P.S Karunarathna – ICT/21/867

Supervisors
Dr. Prabhani Liyanage
Akalanka Panapitiya
Nirasha Kulasooriya

Declaration

I hereby declare that this individual report presents my own contribution to the final year group research project titled “Semantic-Aware Emotional Text-to-Speech for Sinhala: A Single-Speaker Deep Learning Approach.” To the best of my knowledge, this report does not contain any material previously submitted for a degree or diploma in any university, and it does not contain any material written by another person except where due acknowledgement is made.

Declared by: [Your Name]
Date: [Insert Date]

Acknowledgement

This individual report is prepared as part of the final year group research conducted under the Department of Information and Communication Technology, Faculty of Technology, University of Sri Jayewardenepura. I would like to express my sincere gratitude to our supervisors, Dr. Prabhani Liyanage, Akalanka Panapitiya, and Nirasha Kulasooriya, for their guidance, encouragement, and academic support throughout the research process. I also wish to thank my team members for their cooperation and shared effort in completing each stage of the project. Finally, I would like to thank the Department of ICT, the Faculty of Technology, and all others who supported us in carrying out this research successfully.

1. Introduction and Background

This report presents my individual contribution to the group research project titled “Semantic-Aware Emotional Text-to-Speech for Sinhala: A Single-Speaker Deep Learning Approach.” The project focuses on developing a Sinhala emotional text-to-speech system that can generate speech with contextually appropriate emotional prosody based on the meaning of the input text. Unlike conventional Sinhala TTS systems that mainly produce neutral speech, this research aims to create a more natural and expressive system by linking text understanding with emotion-aware speech generation.

The overall research is based on a single-speaker deep learning approach. The main idea is to prepare a suitable Sinhala text and voice dataset, identify emotional meaning from text, and use that information in a speech generation pipeline. The work completed so far includes topic selection, literature review, gap identification, finalization of the problem statement, text data collection, RAG-based synthetic data expansion, manual validation of generated text, voice recording, audio preprocessing, ontology creation, and machine-learning-based emotion classification. At present, the research has moved to the stage of training the final voice model using the prepared dataset.

The research problem addressed by this project is that current Sinhala text-to-speech systems do not properly express the emotional meaning found in text. As a result, the generated speech sounds flat, emotionally sterile, and less natural. From our proposal and presentations, the main gaps identified in this domain are the lack of automated emotion detection for Sinhala text, the absence of culturally appropriate emotional prosody modeling for Sinhala, and the lack of an integrated end-to-end architecture that connects semantic understanding with emotional speech synthesis.

The main objective of the project is to design and develop an intelligent Sinhala emotional text-to-speech system that can automatically generate appropriate emotional prosody from text. The specific objectives include preparing a Sinhala emotional speech and text dataset, developing a semantic-emotion analysis component, creating an intelligent model that can support emotional speech generation, and integrating the overall pipeline into a working research solution.

2. Nature of the Group Work

Although this is an individual report, the research itself was carried out as a group project. Therefore, it is important to explain my contribution in a way that shows both the collaborative nature of the work and my own active participation. The main stages of the research were not completed by one person alone. Instead, the major tasks were divided into sub-parts and shared among the three group members. Because of that, all three members were involved throughout the research process, while each person also took responsibility for specific parts within each stage.

This report is written according to that approach. Rather than claiming that any full stage was completed only by me, I explain how I participated in each major stage, what responsibilities I handled, and how my work supported the shared progress of the project. This is especially important in a research project like ours, where topic selection, methodology development, dataset preparation, ontology design, and model development are strongly interconnected.

3. Individual Contribution by Research Stage

3.1 Topic Selection

The topic selection stage was completed collaboratively by all three team members. Choosing a suitable research topic was not an easy task, because it required us to identify an area that was technically meaningful, relevant to Sinhala language technology, and feasible within our available time and resources. We discussed several possible directions and gradually refined the final topic through collective decision-making.

My contribution at this stage was participating in the discussions that led to the final selection of the topic. I took part in comparing alternatives, thinking about feasibility, and helping the team focus on a research problem that had both academic value and practical relevance. Therefore, although topic selection was a shared activity, I was actively involved in the process that resulted in the final research direction.

3.2 Literature Review

The literature review was one of the earliest and most important stages of the research. In the first phase, the group reviewed approximately thirty-five research papers related to emotional text-to-speech, low-resource language processing, Sinhala TTS, emotion detection, ontology-based semantic modeling, and deep learning approaches for speech synthesis. Since the review workload was shared, I selected and took responsibility for the first ten papers as my individual literature review contribution.

My role was not only to read those ten papers, but also to understand their methods, identify the research gaps they revealed, and prepare summaries that could later support our topic selection, gap identification, methodology design, and technical decisions. Through this review, I gained an understanding of several important areas.

First, some papers helped me understand the importance of text front-end processing such as grapheme-to-phoneme conversion. This was important because Sinhala is a low-resource language and pronunciation-related processing cannot be ignored in TTS development. These studies helped me understand that even before speech generation begins, proper text handling is necessary to achieve good synthesis quality.

Second, I reviewed papers related to low-resource emotional TTS and Sinhala-specific deep learning. These papers helped me understand that data quality, dataset consistency, and careful adaptation are essential when working with low-resource languages. They also showed me that directly applying methods from high-resource languages is often not enough, because language-specific and culture-specific characteristics affect the final performance.

Third, I reviewed papers related to emotion-aware speech synthesis and semantic guidance. These studies showed that emotion should not always be treated as a manually given label. Instead, emotional meaning can be derived from text, context, and semantic cues. This idea strongly supported the direction of our research, where the goal is to detect emotion from text rather than relying only on manual annotation.

Finally, I reviewed papers that showed broader modeling approaches for emotion classification and emotional speech generation. These papers helped me understand how text analysis, emotion detection, prosody control, and synthesis can be combined into a pipeline. This understanding later influenced our ontology design and the creation of a machine learning fallback model.

Overall, my literature review contribution was important because it directly supported later stages of the research. It helped justify the topic, identify the major gaps, and shape the direction of the semantic-emotion analysis component.

3.3 Gap Identification

Gap identification was also handled as a collaborative stage, but my literature review findings directly contributed to this process. Based on the papers I reviewed, I helped support the identification of three major gaps. The first gap was the absence of an automated framework that can derive emotional context from Sinhala text without requiring manual emotion labels. The second gap was the lack of Sinhala-specific emotional prosody modeling. The third gap was the absence of a seamless architecture that links semantic understanding with emotional speech generation.

My contribution at this stage was therefore analytical. By reviewing the first ten papers and summarizing their methods and limitations, I helped provide the evidence needed to justify why this research is necessary and how it differs from existing work.

3.4 Problem Statement Finalization

The problem statement and research objectives were finalized as a group. However, my contribution to the literature review helped support this stage as well. Since the literature clearly showed that Sinhala TTS systems still struggle with emotional expressiveness and semantic awareness, I contributed to shaping the final problem statement around that deficiency.

The finalized problem was that current Sinhala TTS systems cannot properly convey the emotional content embedded in textual meaning, which results in flat and unnatural speech. My contribution to this stage was not separate from the previous stages; rather, it came from helping the team reach a well-supported and researchable problem definition based on the reviewed literature.

3.5 Data Collection

The data collection stage was divided into several parts, and I contributed to multiple sub-tasks within it.

First, I was involved in text data collection. This included helping collect and prepare Sinhala text data that would be suitable for emotional speech synthesis. This was an important stage because the quality and diversity of the text scripts directly affect the usefulness of the final dataset.

Second, the team increased the volume of seed text data using a RAG-based synthetic data generation process. My individual contribution in this stage was manually checking the generated data. This was a necessary task because automatically generated text can contain inconsistencies, irrelevant outputs, unnatural expressions, or emotionally weak content. By manually validating the generated data, I helped improve the quality of the dataset before it was used in later stages.

Third, I participated in the voice recording stage. Since the project follows a single-speaker design, voice consistency was extremely important. My involvement in this stage helped connect the prepared text dataset with the speech data that would later be used for preprocessing and model training.

This stage shows clearly that although data collection was handled as a team effort, my role covered several meaningful sub-parts: text data collection, manual validation of synthetic text, and participation in voice recording.

3.6 Data Preprocessing

The data preprocessing stage became highly important because the project generated a large amount of voice data. Since the dataset was large, the work was shared among the team, and I contributed to all the major preprocessing tasks assigned in this stage.

One of my contributions was adding padding or cropping audio where necessary in order to maintain consistency across the recorded samples. This step was important because inconsistent audio lengths and boundaries can negatively affect training quality.

Another contribution was reducing noise in the recorded audio files. Since speech synthesis models depend heavily on clean and consistent recordings, noise reduction was essential to improve the quality of the training data.

I also contributed to voice enhancement. This was done to improve audio clarity and make the recordings more suitable for model training. Together, these preprocessing tasks helped transform raw recordings into a cleaner and more reliable dataset.

Therefore, my contribution in this stage was not limited to one isolated task. I participated in the practical preparation of the audio dataset by working on consistency, noise reduction, and enhancement.

3.7 Ontology Creation

Ontology creation was a major methodological stage in the research. The team first attempted to create the ontology using a keyword-triggering approach. This stage was carried out through a shared GitHub repository, and I participated in the collaborative development process by contributing through the shared implementation workflow.

However, after testing the initial keyword-triggering approach, it became clear that the method was not accurate enough. It was too limited for handling more complex emotional meaning in Sinhala text. At that point, I carried out an additional literature review focused on ontology creation. I read three more papers related to ontology development, especially frame-based approaches, in order to find a better direction for our research.

This additional review was important because it helped me understand that emotion should not be treated only as a direct keyword match. One of the most important insights I gained at this stage was from frame-based ontology ideas, especially the idea that emotions can be modeled as semantic frames with roles, relations, and contextual interpretation.

Based on this understanding, I suggested that the project should move from simple keyword triggering to a frame-based ontology. This suggestion was later adopted by the team and implemented through the shared GitHub repository. The revised ontology was created according to the literature findings and used a more structured approach involving linguistic analysis, frame matching, and semantic inference. This was a meaningful contribution because it helped improve the conceptual quality and expected accuracy of the emotion detection component.

Therefore, my contribution to ontology creation includes three connected parts: participating in the initial ontology development, identifying the limitations of the keyword-triggering approach, and suggesting the shift to a frame-based ontology after conducting an additional literature review.

3.8 ML Model Creation

The machine learning model creation stage was also developed through the same shared repository. In our research design, the ontology-based method was not treated as the only solution. Instead, the system was designed to include a fallback mechanism so that if the ontology failed, produced a conflict, or gave uncertain output, the system could automatically use a transformer-based machine learning model.

My contribution at this stage was participating in the shared development of this ML component and in the broader idea of using a hybrid approach. This stage was important because it made the system more flexible and practical. Instead of depending only on rule-based or ontology-based reasoning, the model could rely on machine learning when semantic reasoning alone was not enough.

From the latest progress of the research, this stage is presented as a context-aware classification approach using transformer-based embeddings. My contribution was part of the collaborative implementation and design of this fallback logic, which strengthened the emotion detection pipeline of the project.

3.9 Current Ongoing Work: Voice Model Training

At present, the research has moved to the next stage, which is training the final voice model using the prepared voice dataset. In the latest presentation, this is described as the fine-tuning stage of the final speech synthesis model.

My contribution in the current stage is ongoing. I am presently involved in studying how the collected and preprocessed voice data should be used for this model training process. Since this stage is still in progress, it is not yet possible to report final results. However, it is important to state that my role in the research has not ended with earlier tasks. I am still actively involved in the current stage of the project.

4. Reflection on My Overall Contribution

This report shows that my contribution to the research was distributed across the whole project rather than limited to a single part. I participated in the selection of the topic, took responsibility for the first ten papers in the literature review, contributed to the identification of the research gap, helped support the final problem statement, participated in text data collection, manually validated RAG-generated text, took part in voice recording, contributed to audio preprocessing, participated in ontology development, suggested the shift to a frame-based ontology after additional reading, and contributed to the machine learning fallback stage. I am also involved in the current voice-model training stage.

An important point about my contribution is that it reflects the real nature of group research work. In our project, the major stages were shared across the team, but each stage included sub-parts that allowed all members to participate meaningfully. My contribution should therefore be understood as consistent involvement across multiple connected stages of the research process. I was not absent from any major stage. Instead, I contributed through the particular responsibilities I handled within each stage.

5. Conclusion

In conclusion, my contribution to this research extends from the earliest conceptual work to the current ongoing model-training stage. I participated in selecting the research topic, reviewed the first ten papers from the main literature set, contributed to gap identification and problem finalization, took part in text and voice data collection, manually checked synthetic data, worked on audio preprocessing, participated in ontology creation, suggested a more accurate frame-based ontology approach after reviewing additional literature, and contributed to the machine learning fallback model.

These contributions helped the project move from an initial idea to a more structured semantic-aware emotional TTS pipeline for Sinhala. Since the research is still ongoing, especially in the final voice-model training stage, this report represents my contribution so far and reflects both my individual work and my active participation in the shared progress of the group.

Appendix A – Ten Papers Reviewed by Me

1. ElSaadany, O., & Suter, B. (2020) – This paper helped me understand multilingual grapheme-to-phoneme conversion and why pronunciation-related text processing is important for TTS, especially in low-resource languages.

2. Tits, N., El Haddad, K., & Dutoit, T. (2019) – This paper showed how transfer learning can support emotional TTS in low-resource settings and helped me understand how limited data can still be used effectively.

3. Kim, H.-Y., Kim, J.-H., & Kim, J.-M. (2022) – This study provided insight into efficient grapheme-to-phoneme conversion methods and showed how front-end efficiency and accuracy matter in speech systems.

4. Senarath, K.L.P.M. (2024) – This work was directly useful because it focused on Sinhala TTS and highlighted the need for Sinhala-specific deep learning solutions rather than direct reuse of foreign-language models.

5. Abilbekov, A., Mussakhojayeva, S., Yeshpanov, R., & Varol, H.A. (2024) – This paper was useful as an example of emotional TTS dataset creation for a low-resource language and helped me understand good practices for data collection and evaluation.

6. Bott, T., Lux, F., & Vu, N. T. (2024) – This paper showed that emotional control can be guided by natural language prompts, which strengthened the idea that semantic meaning can guide emotional speech generation.

7. Diatlova, D., & Shutov, V. (2023) – This paper helped me understand how FastSpeech2 can be guided toward emotional speech and how emotional intensity may vary across different parts of a sentence.

8. Liu, Y., Wang, L., Gao, S., Yu, Z., Dong, L., & Tian, T. (2025) – This paper provided ideas about cross-lingual representation learning and low-resource speech synthesis, which were useful for understanding broader modeling approaches.

9. Cho, D.-H., Oh, H.-S., Kim, S.-B., Lee, S.-H., & Lee, S.-W. (2024) – This study showed how emotional style and intensity can be modeled more flexibly, and it supported the idea that emotion should be represented in a richer way than simple fixed labels.

10. Dhimate, B. H., Khopade, M. V., Dhere, A. Y., Dhumale, S. D., & Ranjan, N. M. (2021) – This paper was useful because it demonstrated a direct pipeline where text is analyzed for emotion and then converted into emotion-aware speech.

References

Abilbekov, A., Mussakhojayeva, S., Yeshpanov, R., & Varol, H.A. (2024). KazEmoTTS: A dataset for Kazakh emotional text-to-speech synthesis. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024) (pp. 9626–9632).

Bott, T., Lux, F., & Vu, N. T. (2024). Controlling emotion in text-to-speech with natural language prompts. In Proceedings of Interspeech 2024.

Cho, D.-H., Oh, H.-S., Kim, S.-B., Lee, S.-H., & Lee, S.-W. (2024). EmoSphere-TTS: Emotional style and intensity modeling via spherical emotion vector for controllable emotional text-to-speech. arXiv preprint arXiv:2406.07803.

De Giorgis, S., & Gangemi, A. (2024). The Emotion Frame Ontology. arXiv preprint arXiv:2401.10751.

Dhimate, B. H., Khopade, M. V., Dhere, A. Y., Dhumale, S. D., & Ranjan, N. M. (2021). Emotion based text to speech conversion system using recurrent neural network (bi-directional GRU). Journal of Huazhong University of Science and Technology, 50(7).

Diatlova, D., & Shutov, V. (2023). EmoSpeech: Guiding FastSpeech2 towards emotional text to speech. arXiv preprint arXiv:2307.00024.

ElSaadany, O., & Suter, B. (2020). Grapheme-to-phoneme conversion with a multilingual transformer model. Proceedings of the Seventeenth SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, 85–89.

Kim, H.-Y., Kim, J.-H., & Kim, J.-M. (2022). Fast bilingual grapheme-to-phoneme conversion. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Track (pp. 1–10).

Liu, Y., Wang, L., Gao, S., Yu, Z., Dong, L., & Tian, T. (2025). Lao-English code-switched speech synthesis via neural codec language modeling. In Proceedings of the 24th China National Conference on Computational Linguistics (pp. 1067–1077).

Senarath, K.L.P.M. (2024). Enhancing Sinhala text-to-speech system using deep learning techniques. Dissertation, Master of Computer Science, University of Colombo School of Computing.

Tits, N., El Haddad, K., & Dutoit, T. (2019). Exploring transfer learning for low resource emotional TTS. arXiv preprint arXiv:1901.04276v1.

I can also turn this into a shorter, more formal university-style version if you want.
