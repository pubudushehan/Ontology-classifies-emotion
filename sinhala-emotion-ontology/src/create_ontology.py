from rdflib import Graph, Literal, RDF, RDFS, Namespace, URIRef
import json
import os

# Define Namespaces
SEO = Namespace("http://www.semanticweb.org/sinhala-emotion-ontology#")
EX = Namespace("http://example.org/")

def create_ontology():
    g = Graph()
    g.bind("seo", SEO)
    
    # Define Classes
    emotion_class = SEO.Emotion
    word_class = SEO.Word
    
    g.add((emotion_class, RDF.type, RDFS.Class))
    g.add((word_class, RDF.type, RDFS.Class))
    
    # Define Properties
    has_emotion = SEO.hasEmotion
    g.add((has_emotion, RDF.type, RDF.Property))
    g.add((has_emotion, RDFS.domain, word_class))
    g.add((has_emotion, RDFS.range, emotion_class))

    # Load Lexicon
    try:
        with open("ontology/lexicon.json", "r", encoding="utf-8") as f:
            lexicon = json.load(f)
    except FileNotFoundError:
        print("Error: ontology/lexicon.json not found. Run from project root.")
        return

    # Populate Ontology
    for emotion, keywords in lexicon.items():
        emotion_uri = SEO[emotion]
        g.add((emotion_uri, RDF.type, emotion_class))
        g.add((emotion_uri, RDFS.label, Literal(emotion)))
        
        for word in keywords:
            # Create a safe URI for the word (encoding might be needed for complex scripts, but URIRef handles basics)
            # Using a hash or simple counter might be safer for non-ASCII URIs in some stores, 
            # but for this demo, we'll try to encode it or just use an ID.
            # Let's use an ID to be safe and add the label.
            word_id = f"word_{hash(word)}"
            word_uri = SEO[word_id]
            
            g.add((word_uri, RDF.type, word_class))
            g.add((word_uri, RDFS.label, Literal(word, lang="si")))
            g.add((word_uri, has_emotion, emotion_uri))

    # Serialize
    output_path = "ontology/sinhala_emotion.ttl"
    g.serialize(destination=output_path, format="turtle")
    print(f"Ontology generated at {output_path}")

    # Verification (SPARQL)
    query = """
    SELECT ?word ?emotion
    WHERE {
        ?w rdf:type seo:Word .
        ?w rdfs:label ?word .
        ?w seo:hasEmotion ?e .
        ?e rdfs:label ?emotion .
    } LIMIT 100
    """
    
    print("\nVerifying with SPARQL (First 100 entries):")
    results = g.query(query)
    for row in results:
        print(f"{row.word} -> {row.emotion}")

if __name__ == "__main__":
    # Ensure we run from the correct directory or adjust paths
    if not os.path.exists("ontology"):
        os.makedirs("ontology")
    create_ontology()
