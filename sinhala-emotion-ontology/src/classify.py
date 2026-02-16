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

    def classify_ontology(self, text):
        """
        Check if any token in the text exists in the ontology as a word with an emotion.
        """
        tokens = self.tokenize(text)
        
        # Simple keyword matching against ontology
        # Query: ?w label "token" . ?w hasEmotion ?e . ?e label ?emotion
        
        for token in tokens:
            token_literal = Literal(token, lang="si")
            # Try to match label. Note: Our ontology has labels as literals.
            # We iterate or query. Query is cleaner but might be slower if many tokens.
            # Let's use a query for the whole text tokens if possible, or just iterate.
            # Iterating tokens and query for each:
            
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
            
            # Use binding
            # Note: rdflib might need precise literal matching (lang tag etc)
            # In create_ontology.py we added lang="si".
            # So we must match that.
            
            # Let's try to query.
            # This is a basic implementation. A more robust one would gather all hits and vote.
            
            # Optimization: Load lexicon into memory map for fast lookup if ontology is large.
            # But here we stick to "Using Ontology".
            
            res = self.g.query(query, initBindings={'target': Literal(token)}) # try without lang first? No, sparql filter str() works on value.
            
            for row in res:
                return str(row.emotion_label), 1.0 # High confidence for lexicon match
                
            # Try with lang tag if above fails or just iterate all words in graph (slower but safer)
            # Actually, let's keep it simple. If SPARQL fails, we fall back to ML.
            
        return None, 0.0

    def classify_ml(self, text):
        if not self.model or not self.centroids:
            return "Unknown", 0.0

        embedding = self.model.encode(text)
        embedding = embedding / np.linalg.norm(embedding)

        best_label = "Neutral"
        best_score = -1.0
        
        for label, centroid in self.centroids.items():
            if label == "Neutral": continue # We handle Neutral via threshold usually, but let's see. 
            # Actually, if we have a Neutral centroid (we have 1 sample), it might be bad.
            # So let's ignore the single Neutral sample centroid if it exists, or just use it?
            # 1 sample is not enough to form a cluster.
            # Strategy: Compare with Happy, Sad, Angry.
            # If max score < 0.6 -> Neutral.
            
            score = np.dot(embedding, centroid)
            if score > best_score:
                best_score = score
                best_label = label
                
        # Threshold for Neutral
        if best_score < 0.25: # LaBSE is cosine sim. 0.25 is very low. 
            # In practice, even different sentences might have 0.3-0.4 similarity.
            # Let's ensure we pick the *best* one, and if it's really low, Neutral.
            # But the user wants "Neutral" classification too.
            # Since we lack neutral training data, we assume "None of the above" = Neutral.
            return "Neutral", round(float(best_score), 4) # Return Neutral but with the score of the closest other emotion? Or just 0.5?
            # Let's return Neutral and low confidence.
        
        return best_label, round(float(best_score), 4)

    def predict(self, text):
        # 1. Ontology Check
        label, conf = self.classify_ontology(text)
        if label:
            return {"label": label, "confidence": conf, "method": "Ontology"}
        
        # 2. ML Check
        label, conf = self.classify_ml(text)
        return {"label": label, "confidence": conf, "method": "ML (LaBSE)"}

if __name__ == "__main__":
    # Test
    classifier = EmotionClassifier()
    test_sentences = [
        "මම අද ගොඩක් සතුටින්",  # Expect Happy
        "මට දුකයි",            # Expect Sad
        "මම කේන්තියෙන් ඉන්නේ",      # Expect Angry (if lexicon matches or ML)
    ]
    for t in test_sentences:
        print(f"Text: {t} -> {classifier.predict(t)}")
