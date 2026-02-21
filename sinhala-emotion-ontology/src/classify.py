import json
import os
import pickle
import numpy as np
from rdflib import Graph, Namespace, RDF, RDFS, Literal
from sentence_transformers import SentenceTransformer
from indicnlp.tokenize import indic_tokenize

# Define Namespaces
SEO = Namespace("http://www.semanticweb.org/sinhala-emotion-ontology#")

class EmotionClassifier:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ontology_path = os.path.join(self.base_dir, "ontology", "sinhala_emotion.ttl")
        self.centroids_path = os.path.join(self.base_dir, "data", "centroids.pkl")
        
        # Load Ontology
        self.g = Graph()
        if os.path.exists(self.ontology_path):
            self.g.parse(self.ontology_path, format="turtle")
            print("Ontology loaded.")
        else:
            print("Warning: Ontology file not found.")

        # Load Centroids
        self.centroids = {}
        if os.path.exists(self.centroids_path):
            with open(self.centroids_path, 'rb') as f:
                self.centroids = pickle.load(f)
            print("Centroids loaded.")
        else:
            print("Warning: Centroids file not found. ML classification will be disabled.")

        # Load Model
        # optimization: load strictly if needed or load once.
        # Since this is a service, load once.
        try:
            self.model = SentenceTransformer("sentence-transformers/LaBSE")
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def tokenize(self, text):
        return indic_tokenize.trivial_tokenize(text)
        return emotion_counts, matched_words_dict

    def classify_ontology(self, text):
        """
        Check all tokens in the text against the ontology.
        Returns:
            - emotion_counts: Dict of emotion counts (e.g., {'Happy': 2})
            - matched_words: Dict of list of words per emotion (e.g., {'Happy': ['word1', 'word2']})
        """
        tokens = self.tokenize(text)
        emotion_counts = {}
        matched_words_dict = {}
        
        for token in tokens:
            # SPARQL Query to find emotion for the token
            query = """
            SELECT ?emotion_label
            WHERE {
                ?w rdf:type seo:Word .
                ?w rdfs:label ?label .
                ?w seo:hasEmotion ?e .
                ?e rdfs:label ?emotion_label .
                FILTER(str(?label) = ?target)
            }
            """
            
            # Execute query
            res = self.g.query(query, initBindings={'target': Literal(token)})
            
            for row in res:
                emotion = str(row.emotion_label)
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                    matched_words_dict[emotion].append(token)
                else:
                    emotion_counts[emotion] = 1
                    matched_words_dict[emotion] = [token]
                    
        return emotion_counts, matched_words_dict

    def classify_ml(self, text):
        if not self.model or not self.centroids:
            return "Unknown", 0.0

        embedding = self.model.encode(text)
        embedding = embedding / np.linalg.norm(embedding)

        best_label = "Neutral"
        best_score = -1.0
        
        for label, centroid in self.centroids.items():
            if label == "Neutral": continue 
            
            score = np.dot(embedding, centroid)
            if score > best_score:
                best_score = score
                best_label = label
                
        # Threshold for Neutral
        if best_score < 0.25: 
            return "Neutral", round(float(best_score), 4)
        
        return best_label, round(float(best_score), 4)

    def predict(self, text):
        # 1. Ontology Check (All words)
        emotion_counts, matched_words = self.classify_ontology(text)
        
        # Logic:
        # - If no matches -> ML
        # - If matches found for ONLY ONE emotion -> Return that emotion (Ontology)
        # - If matches found for MULTIPLE emotions -> Conflict -> ML
        
        if not emotion_counts:
            # No ontology matches
            label, conf = self.classify_ml(text)
            return {
                "label": label, 
                "confidence": conf, 
                "method": "ML (LaBSE) - No Ontology Match",
                "matched_words": {}
            }
            
        found_emotions = list(emotion_counts.keys())
        
        if len(found_emotions) == 1:
            # Single emotion matched (clean match)
            emotion = found_emotions[0]
            count = emotion_counts[emotion]
            # Confidence could be 1.0 or scaled by count? 1.0 for now as it's rule-based.
            return {
                "label": emotion, 
                "confidence": 1.0, 
                "method": f"Ontology (Matched {count} words)",
                "matched_words": matched_words
            }
            
        else:
            # Conflict (e.g. {'Happy': 1, 'Sad': 1})
            # Fallback to ML to resolve context
            label, conf = self.classify_ml(text)
            return {
                "label": label, 
                "confidence": conf, 
                "method": f"ML (LaBSE) - Conflict Resolution {emotion_counts}",
                "matched_words": matched_words
            }

if __name__ == "__main__":
    # Test
    classifier = EmotionClassifier()
    test_sentences = [
        "මම අද ගොඩක් සතුටුයි",  # Expect Happy (matches 'සතුටුයි')
        "මට දුකයි",            # Expect Sad
        "මම කේන්තියෙන් ඉන්නේ",      # Expect Angry (if lexicon matches or ML)
    ]
    for t in test_sentences:
        print(f"Text: {t} -> {classifier.predict(t)}")
