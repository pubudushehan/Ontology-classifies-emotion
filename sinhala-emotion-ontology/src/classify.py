"""
classify.py -- 3-Tier Semantic Frame-Based Emotion Classifier for Sinhala

Tier 1 (Linguistic Analysis):
    Detect negation, intensifiers, diminishers, discourse connectives,
    and semantic roles (agent/patient/experiencer) via Sinhala case markers.

Tier 2 (Frame-Based Ontology Matching):
    Match tokens to EmotionFrames (not directly to emotions) via SPARQL.
    Each frame encodes typicalEmotion, agentEmotion, patientEmotion,
    negatedEmotion, polarity, and weight.

Tier 3 (Semantic Inference):
    Combine frame matches with linguistic context:
    - Negation flips to negatedEmotion
    - Role markers select agent/patient emotion variant
    - Intensifiers boost confidence, diminishers reduce it
    - Contrastive discourse connectives weight the post-connective clause
    - Hostile address forms add anger signal

The ML fallback (LaBSE centroid similarity) is unchanged.
"""

import json
import os
import pickle
import numpy as np
from rdflib import Graph, Namespace, RDF, RDFS, Literal
from sentence_transformers import SentenceTransformer
from indicnlp.tokenize import indic_tokenize

# Define Namespaces
SEO = Namespace("http://www.semanticweb.org/sinhala-emotion-ontology#")

# ---------------------------------------------------------------------------
# Constants for Tier 1 analysis
# ---------------------------------------------------------------------------
NEGATION_WINDOW = 2      # max token distance for negation to affect a frame
INTENSIFIER_WINDOW = 3   # max token distance for intensifier to affect a frame
HOSTILE_ADDRESS_WEIGHT = 0.7  # anger signal added when hostile pronouns detected


class EmotionClassifier:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ontology_path = os.path.join(self.base_dir, "ontology", "sinhala_emotion.ttl")
        self.centroids_path = os.path.join(self.base_dir, "data", "centroids.pkl")
        self.frames_path = os.path.join(self.base_dir, "ontology", "frames.json")
        self.modifiers_path = os.path.join(self.base_dir, "ontology", "modifiers.json")
        self.role_markers_path = os.path.join(self.base_dir, "ontology", "role_markers.json")

        # ----------------------------------------------------------
        # Load Ontology (RDF graph)
        # ----------------------------------------------------------
        self.g = Graph()
        if os.path.exists(self.ontology_path):
            self.g.parse(self.ontology_path, format="turtle")
            print("Ontology loaded.")
        else:
            print("Warning: Ontology file not found.")

        # ----------------------------------------------------------
        # Load semantic data files
        # ----------------------------------------------------------
        self.frames_data = {}
        if os.path.exists(self.frames_path):
            with open(self.frames_path, 'r', encoding='utf-8') as f:
                self.frames_data = json.load(f)
            print(f"Frames loaded: {len(self.frames_data)} frames.")

        self.modifiers_data = {}
        if os.path.exists(self.modifiers_path):
            with open(self.modifiers_path, 'r', encoding='utf-8') as f:
                self.modifiers_data = json.load(f)
            print("Modifiers loaded.")

        self.role_markers_data = {}
        if os.path.exists(self.role_markers_path):
            with open(self.role_markers_path, 'r', encoding='utf-8') as f:
                self.role_markers_data = json.load(f)
            print("Role markers loaded.")

        # ----------------------------------------------------------
        # Build quick-lookup structures
        # ----------------------------------------------------------
        self._build_modifier_lookups()
        self._build_role_marker_lookups()

        # ----------------------------------------------------------
        # Load Centroids (ML)
        # ----------------------------------------------------------
        self.centroids = {}
        if os.path.exists(self.centroids_path):
            with open(self.centroids_path, 'rb') as f:
                self.centroids = pickle.load(f)
            print("Centroids loaded.")
        else:
            print("Warning: Centroids file not found. ML classification will be disabled.")

        # ----------------------------------------------------------
        # Load Sentence Transformer (ML)
        # ----------------------------------------------------------
        try:
            self.model = SentenceTransformer("sentence-transformers/LaBSE")
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    # ===================================================================
    # Initialization helpers
    # ===================================================================
    def _build_modifier_lookups(self):
        """Build O(1) lookup sets for negation, intensifiers, diminishers, connectives."""
        mod = self.modifiers_data

        # Negation set
        self.negation_words = set(mod.get("negation", {}).get("words", []))

        # Intensifiers: word -> multiplier
        self.intensifier_map = {}
        for level, level_data in mod.get("intensifiers", {}).get("levels", {}).items():
            multiplier = level_data.get("multiplier", 1.0)
            for word in level_data.get("words", []):
                self.intensifier_map[word] = multiplier

        # Diminishers: word -> multiplier
        dim_multiplier = mod.get("diminishers", {}).get("multiplier", 0.6)
        self.diminisher_map = {}
        for word in mod.get("diminishers", {}).get("words", []):
            self.diminisher_map[word] = dim_multiplier

        # Discourse connectives: word -> (type, pre_weight, post_weight)
        self.connective_map = {}
        for conn_type, conn_data in mod.get("discourse_connectives", {}).get("types", {}).items():
            pre_w = conn_data.get("pre_clause_weight", 1.0)
            post_w = conn_data.get("post_clause_weight", 1.0)
            for word in conn_data.get("words", []):
                self.connective_map[word] = (conn_type, pre_w, post_w)

    def _build_role_marker_lookups(self):
        """Build O(1) lookup sets for role detection and hostile address detection."""
        rm = self.role_markers_data

        # All markers: word -> role
        self.role_word_map = {}  # word -> "agent" | "patient" | "experiencer" | "possessive"
        self.hostile_words = set()
        self.first_person_agent = set()
        self.first_person_patient = set()
        self.first_person_experiencer = set()

        for marker_type, marker_info in rm.items():
            role = marker_info.get("role", "")
            for group_name, group_data in marker_info.get("groups", {}).items():
                is_hostile = group_data.get("hostile", False)
                words = group_data.get("words", [])
                for word in words:
                    self.role_word_map[word] = role
                    if is_hostile:
                        self.hostile_words.add(word)
                    if "first_person" in group_name:
                        if role == "agent":
                            self.first_person_agent.add(word)
                        elif role == "patient":
                            self.first_person_patient.add(word)
                        elif role == "experiencer":
                            self.first_person_experiencer.add(word)

    # ===================================================================
    # Tokenization
    # ===================================================================
    def tokenize(self, text):
        return indic_tokenize.trivial_tokenize(text)

    # ===================================================================
    # Tier 1: Linguistic Analysis
    # ===================================================================
    def _analyze_linguistics(self, tokens):
        """
        Scan tokens for negation, intensifiers, diminishers, connectives, and roles.

        Returns a dict with:
            negation_positions:     list of token indices
            intensifier_positions:  list of (index, multiplier)
            diminisher_positions:   list of (index, multiplier)
            connective_positions:   list of (index, type, pre_weight, post_weight)
            role_info: {
                speaker_is_agent:       bool
                speaker_is_patient:     bool
                speaker_is_experiencer: bool
                hostile_address:        bool
                hostile_count:          int
            }
        """
        negation_positions = []
        intensifier_positions = []
        diminisher_positions = []
        connective_positions = []

        speaker_is_agent = False
        speaker_is_patient = False
        speaker_is_experiencer = False
        hostile_address = False
        hostile_count = 0

        for idx, token in enumerate(tokens):
            # Negation
            if token in self.negation_words:
                negation_positions.append(idx)

            # Intensifiers
            if token in self.intensifier_map:
                intensifier_positions.append((idx, self.intensifier_map[token]))

            # Diminishers
            if token in self.diminisher_map:
                diminisher_positions.append((idx, self.diminisher_map[token]))

            # Discourse connectives
            if token in self.connective_map:
                conn_type, pre_w, post_w = self.connective_map[token]
                connective_positions.append((idx, conn_type, pre_w, post_w))

            # Role markers
            if token in self.first_person_agent:
                speaker_is_agent = True
            if token in self.first_person_patient:
                speaker_is_patient = True
            if token in self.first_person_experiencer:
                speaker_is_experiencer = True
            if token in self.hostile_words:
                hostile_address = True
                hostile_count += 1

            # Check for verb-final negation forms (e.g., "කරන්නෑ" = doesn't do)
            # Only check for "නෑ"/"බෑ" suffixes on tokens > 5 chars to avoid
            # false positives on short words like "නෑනෙ" (sister-in-law)
            if token not in self.negation_words and len(token) > 5:
                for neg_suffix in ["න්නෑ", "න්නැහැ", "න්බෑ", "න්බැහැ"]:
                    if token.endswith(neg_suffix):
                        negation_positions.append(idx)
                        break

        return {
            "negation_positions": negation_positions,
            "intensifier_positions": intensifier_positions,
            "diminisher_positions": diminisher_positions,
            "connective_positions": connective_positions,
            "role_info": {
                "speaker_is_agent": speaker_is_agent,
                "speaker_is_patient": speaker_is_patient,
                "speaker_is_experiencer": speaker_is_experiencer,
                "hostile_address": hostile_address,
                "hostile_count": hostile_count,
            }
        }

    # ===================================================================
    # Tier 2: Frame-Based Ontology Matching
    # ===================================================================
    def _match_frames(self, tokens):
        """
        For each token, query the ontology for EmotionFrame matches.
        Includes length-constrained prefix matching and per-token deduplication.

        Returns a list of match dicts:
            {token_idx, token, frame_name, typicalEmotion, agentEmotion,
             patientEmotion, negatedEmotion, polarity, weight}
        """
        MAX_PREFIX_DIFF = 3  # max char difference for prefix matching

        raw_matches = []  # collect all, then deduplicate

        for idx, token in enumerate(tokens):
            if len(token) < 3:
                continue

            # Skip tokens that are modifiers (negation, intensifier, etc.)
            # to avoid them also matching as frame triggers
            if (token in self.negation_words or token in self.intensifier_map
                    or token in self.diminisher_map or token in self.connective_map):
                continue

            query = """
            SELECT ?frame_label ?label
            WHERE {
                ?w rdf:type seo:LexicalTrigger .
                ?w rdfs:label ?label .
                ?w seo:triggersFrame ?f .
                ?f rdfs:label ?frame_label .
                FILTER(
                    str(?label) = ?target ||
                    STRSTARTS(str(?label), ?target) ||
                    STRSTARTS(?target, str(?label))
                )
            }
            """
            res = self.g.query(query, initBindings={'target': Literal(token)})

            for row in res:
                label_str = str(row.label)
                frame_name = str(row.frame_label)

                # Length-constrained prefix matching:
                # For non-exact matches, reject if length difference > MAX_PREFIX_DIFF
                if label_str != token:
                    if abs(len(label_str) - len(token)) > MAX_PREFIX_DIFF:
                        continue

                frame_props = self.frames_data.get(frame_name, {})
                if not frame_props:
                    continue

                raw_matches.append({
                    "token_idx": idx,
                    "token": token,
                    "matched_label": label_str,
                    "frame_name": frame_name,
                    "typicalEmotion": frame_props.get("typicalEmotion", "Neutral"),
                    "agentEmotion": frame_props.get("agentEmotion", "Neutral"),
                    "patientEmotion": frame_props.get("patientEmotion", "Neutral"),
                    "negatedEmotion": frame_props.get("negatedEmotion", "Neutral"),
                    "polarity": frame_props.get("polarity", "neutral"),
                    "weight": frame_props.get("weight", 0.5),
                })

        # Deduplicate: keep only one match per (token_idx, frame_name),
        # preferring the most specific match (longest matched_label)
        best = {}
        for m in raw_matches:
            key = (m["token_idx"], m["frame_name"])
            if key not in best or len(m["matched_label"]) > len(best[key]["matched_label"]):
                best[key] = m

        return list(best.values())

    # ===================================================================
    # Tier 3: Semantic Inference
    # ===================================================================
    def _is_negated(self, token_idx, negation_positions):
        """Check if token is within negation scope (within NEGATION_WINDOW tokens)."""
        for neg_idx in negation_positions:
            if abs(token_idx - neg_idx) <= NEGATION_WINDOW and token_idx != neg_idx:
                return True
        return False

    def _get_intensifier_multiplier(self, token_idx, intensifier_positions):
        """Get the highest intensifier multiplier within window, or 1.0 if none."""
        best = 1.0
        for int_idx, multiplier in intensifier_positions:
            if abs(token_idx - int_idx) <= INTENSIFIER_WINDOW and token_idx != int_idx:
                best = max(best, multiplier)
        return best

    def _get_diminisher_multiplier(self, token_idx, diminisher_positions):
        """Get the lowest diminisher multiplier within window, or 1.0 if none."""
        best = 1.0
        for dim_idx, multiplier in diminisher_positions:
            if abs(token_idx - dim_idx) <= INTENSIFIER_WINDOW and token_idx != dim_idx:
                best = min(best, multiplier)
        return best

    def _get_discourse_weight(self, token_idx, connective_positions):
        """Determine discourse weight for this token based on connective positions."""
        weight = 1.0
        for conn_idx, conn_type, pre_w, post_w in connective_positions:
            if conn_type == "contrastive":
                if token_idx < conn_idx:
                    weight *= pre_w   # Pre-connective clause: de-weighted
                elif token_idx > conn_idx:
                    weight *= post_w  # Post-connective clause: boosted
        return weight

    def _infer_emotions(self, frame_matches, linguistic_context, tokens):
        """
        Combine frame matches with linguistic context to produce final emotion scores.

        Returns:
            emotion_scores:  dict {emotion: weighted_score}
            matched_words:   dict {emotion: [words]}
            explanation:     list of inference steps (for debugging/display)
        """
        emotion_scores = {}
        matched_words = {}
        explanation = []

        role_info = linguistic_context["role_info"]
        neg_positions = linguistic_context["negation_positions"]
        int_positions = linguistic_context["intensifier_positions"]
        dim_positions = linguistic_context["diminisher_positions"]
        conn_positions = linguistic_context["connective_positions"]

        for match in frame_matches:
            token_idx = match["token_idx"]
            frame_name = match["frame_name"]
            token = match["token"]

            # ----- Step A: Determine base emotion from role -----
            if role_info["speaker_is_patient"]:
                emotion = match["patientEmotion"]
                role_used = "patient"
            elif role_info["speaker_is_agent"]:
                emotion = match["agentEmotion"]
                role_used = "agent"
            else:
                emotion = match["typicalEmotion"]
                role_used = "typical"

            # ----- Step B: Check negation -----
            # Only flip polarity for strong frames (weight >= 0.7).
            # Weak contextual frames are not reliably negated by nearby particles.
            negated = False
            if match["weight"] >= 0.7:
                negated = self._is_negated(token_idx, neg_positions)
            if negated:
                emotion = match["negatedEmotion"]
                explanation.append(f"'{token}' [{frame_name}] negated -> {emotion}")
            else:
                explanation.append(f"'{token}' [{frame_name}] role={role_used} -> {emotion}")

            # ----- Step C: Compute weight -----
            weight = match["weight"]

            # Intensifier boost
            int_mult = self._get_intensifier_multiplier(token_idx, int_positions)
            weight *= int_mult
            if int_mult > 1.0:
                explanation.append(f"  intensifier boost x{int_mult}")

            # Diminisher reduction
            dim_mult = self._get_diminisher_multiplier(token_idx, dim_positions)
            weight *= dim_mult
            if dim_mult < 1.0:
                explanation.append(f"  diminisher reduction x{dim_mult}")

            # Discourse weight
            disc_w = self._get_discourse_weight(token_idx, conn_positions)
            weight *= disc_w
            if disc_w != 1.0:
                explanation.append(f"  discourse weight x{disc_w}")

            # ----- Step D: Accumulate -----
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.0
                matched_words[emotion] = []
            emotion_scores[emotion] += weight
            matched_words[emotion].append(token)

        # ----- Step E: Hostile address anger signal -----
        if role_info["hostile_address"]:
            hostile_signal = HOSTILE_ADDRESS_WEIGHT * role_info["hostile_count"]
            if "Angry" not in emotion_scores:
                emotion_scores["Angry"] = 0.0
                matched_words["Angry"] = []
            emotion_scores["Angry"] += hostile_signal
            explanation.append(
                f"Hostile address detected ({role_info['hostile_count']}x) -> Angry +{hostile_signal:.2f}"
            )

        return emotion_scores, matched_words, explanation

    # ===================================================================
    # Combined Ontology Classification (3-Tier)
    # ===================================================================
    def classify_ontology(self, text):
        """
        Full 3-tier semantic ontology classification.

        Returns:
            emotion_scores:  dict {emotion: weighted_score}
            matched_words:   dict {emotion: [words]}
            explanation:     list of inference step strings
        """
        tokens = self.tokenize(text)

        # Tier 1: Linguistic Analysis
        linguistic_context = self._analyze_linguistics(tokens)

        # Tier 2: Frame Matching
        frame_matches = self._match_frames(tokens)

        # Tier 3: Semantic Inference
        emotion_scores, matched_words, explanation = self._infer_emotions(
            frame_matches, linguistic_context, tokens
        )

        return emotion_scores, matched_words, explanation

    # ===================================================================
    # ML Classification (unchanged)
    # ===================================================================
    def classify_ml(self, text):
        if not self.model or not self.centroids:
            return "Unknown", 0.0

        embedding = self.model.encode(text)
        embedding = embedding / np.linalg.norm(embedding)

        best_label = "Neutral"
        best_score = -1.0

        for label, centroid in self.centroids.items():
            if label == "Neutral":
                continue

            score = np.dot(embedding, centroid)
            if score > best_score:
                best_score = score
                best_label = label

        # Threshold for Neutral
        if best_score < 0.25:
            return "Neutral", round(float(best_score), 4)

        return best_label, round(float(best_score), 4)

    # ===================================================================
    # Hybrid Prediction
    # ===================================================================
    def predict(self, text):
        # 1. Ontology Check (3-Tier Semantic)
        emotion_scores, matched_words, explanation = self.classify_ontology(text)

        # Logic:
        # - If no matches -> ML
        # - If matches found for ONLY ONE emotion -> Return that emotion (Ontology)
        # - If matches found for MULTIPLE emotions -> pick dominant if clear, else ML

        if not emotion_scores:
            # No ontology matches
            label, conf = self.classify_ml(text)
            return {
                "label": label,
                "confidence": conf,
                "method": "ML (LaBSE) - No Ontology Match",
                "matched_words": {},
                "explanation": ["No frames matched -> fallback to ML"]
            }

        found_emotions = list(emotion_scores.keys())

        if len(found_emotions) == 1:
            # Single emotion matched (clean match)
            emotion = found_emotions[0]
            total_weight = emotion_scores[emotion]
            # Confidence: scale weight to [0, 1] range
            confidence = min(round(total_weight / 2.0, 4), 1.0)
            confidence = max(confidence, 0.5)  # Ontology match is at least 0.5 conf
            return {
                "label": emotion,
                "confidence": confidence,
                "method": f"Ontology (Frame-based, {len(matched_words.get(emotion, []))} triggers)",
                "matched_words": matched_words,
                "explanation": explanation
            }

        else:
            # Multiple emotions detected -- check if one clearly dominates
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            top_emotion, top_score = sorted_emotions[0]
            second_emotion, second_score = sorted_emotions[1]

            # If top emotion has >2x the score of second, it dominates
            if top_score > 0 and (second_score == 0 or top_score / second_score >= 2.0):
                confidence = min(round(top_score / (top_score + second_score), 4), 1.0)
                return {
                    "label": top_emotion,
                    "confidence": confidence,
                    "method": f"Ontology (Frame-based, dominant: {top_emotion}={top_score:.2f} vs {second_emotion}={second_score:.2f})",
                    "matched_words": matched_words,
                    "explanation": explanation
                }
            else:
                # True conflict -- fallback to ML
                label, conf = self.classify_ml(text)
                return {
                    "label": label,
                    "confidence": conf,
                    "method": f"ML (LaBSE) - Frame Conflict {dict(sorted_emotions)}",
                    "matched_words": matched_words,
                    "explanation": explanation + [
                        f"Frame conflict: {dict(sorted_emotions)} -> fallback to ML"
                    ]
                }


if __name__ == "__main__":
    # Test
    classifier = EmotionClassifier()
    test_sentences = [
        "මම අද ගොඩක් සතුටුයි",          # Expect Happy
        "මට දුකයි",                        # Expect Sad
        "මම කේන්තියෙන් ඉන්නේ",            # Expect Angry (ML likely)
        "මම සතුටු නෑ",                     # Expect Sad (negation!)
        "හරිම දුකයි",                      # Expect Sad (intensified)
        "පලයන් යකෝ මෙතනින්",              # Expect Angry (threat + exclamation)
        "සතුටු වුණත් දුකයි",              # Expect Sad (contrastive: but sad)
        "තෝ මොකෙක්ද යකෝ",                # Expect Angry (hostile address)
    ]
    for t in test_sentences:
        result = classifier.predict(t)
        print(f"\nText: {t}")
        print(f"  Label: {result['label']}, Confidence: {result['confidence']}")
        print(f"  Method: {result['method']}")
        if result.get('explanation'):
            for step in result['explanation']:
                print(f"    -> {step}")
