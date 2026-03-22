"""
create_ontology.py -- Generate frame-based Sinhala emotion ontology (RDF/Turtle)

This replaces the old flat keyword-to-emotion mapping with a 3-tier semantic
frame-based ontology inspired by:
  - Emotion Frame Ontology (EFO, De Giorgis & Gangemi 2024)
  - TONE 3-Tiered Ontology for Emotion (2024)

Schema:
  LexicalTrigger --triggersFrame--> EmotionFrame
  EmotionFrame   --hasTypicalEmotion--> Emotion
  EmotionFrame   --hasAgentEmotion-->   Emotion
  EmotionFrame   --hasPatientEmotion--> Emotion
  EmotionFrame   --hasNegatedEmotion--> Emotion
  EmotionFrame   --hasPolarity-->       xsd:string
  EmotionFrame   --hasWeight-->         xsd:float

Modifiers (NegationMarker, Intensifier, DiscourseConnective) are stored in the
TTL for completeness but are also loaded directly from JSON at classify time.
"""

from rdflib import Graph, Literal, RDF, RDFS, Namespace, URIRef, XSD
import json
import os

# Define Namespaces
SEO = Namespace("http://www.semanticweb.org/sinhala-emotion-ontology#")


def create_ontology():
    g = Graph()
    g.bind("seo", SEO)
    g.bind("xsd", XSD)

    # ------------------------------------------------------------------ #
    #  1. Define Classes
    # ------------------------------------------------------------------ #
    emotion_class = SEO.Emotion
    emotion_frame_class = SEO.EmotionFrame
    lexical_trigger_class = SEO.LexicalTrigger
    negation_marker_class = SEO.NegationMarker
    intensifier_class = SEO.Intensifier
    diminisher_class = SEO.Diminisher
    discourse_connective_class = SEO.DiscourseConnective

    for cls in [emotion_class, emotion_frame_class, lexical_trigger_class,
                negation_marker_class, intensifier_class, diminisher_class,
                discourse_connective_class]:
        g.add((cls, RDF.type, RDFS.Class))

    # ------------------------------------------------------------------ #
    #  2. Define Properties
    # ------------------------------------------------------------------ #
    # Frame-trigger relationship
    triggers_frame = SEO.triggersFrame
    g.add((triggers_frame, RDF.type, RDF.Property))
    g.add((triggers_frame, RDFS.domain, lexical_trigger_class))
    g.add((triggers_frame, RDFS.range, emotion_frame_class))

    # Frame-to-Emotion properties
    has_typical_emotion = SEO.hasTypicalEmotion
    has_agent_emotion = SEO.hasAgentEmotion
    has_patient_emotion = SEO.hasPatientEmotion
    has_negated_emotion = SEO.hasNegatedEmotion

    for prop in [has_typical_emotion, has_agent_emotion,
                 has_patient_emotion, has_negated_emotion]:
        g.add((prop, RDF.type, RDF.Property))
        g.add((prop, RDFS.domain, emotion_frame_class))
        g.add((prop, RDFS.range, emotion_class))

    # Frame metadata
    has_polarity = SEO.hasPolarity
    has_weight = SEO.hasWeight
    g.add((has_polarity, RDF.type, RDF.Property))
    g.add((has_weight, RDF.type, RDF.Property))

    # Modifier properties
    has_intensity_level = SEO.hasIntensityLevel
    has_effect = SEO.hasEffect
    g.add((has_intensity_level, RDF.type, RDF.Property))
    g.add((has_effect, RDF.type, RDF.Property))

    # ------------------------------------------------------------------ #
    #  3. Resolve paths relative to project root
    # ------------------------------------------------------------------ #
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_path = os.path.join(base_dir, "ontology", "frames.json")
    modifiers_path = os.path.join(base_dir, "ontology", "modifiers.json")
    output_path = os.path.join(base_dir, "ontology", "sinhala_emotion.ttl")

    # ------------------------------------------------------------------ #
    #  4. Load frames data
    # ------------------------------------------------------------------ #
    try:
        with open(frames_path, "r", encoding="utf-8") as f:
            frames = json.load(f)
    except FileNotFoundError:
        print(f"Error: {frames_path} not found.")
        return

    # ------------------------------------------------------------------ #
    #  5. Create Emotion instances
    # ------------------------------------------------------------------ #
    emotion_set = set()
    for frame_data in frames.values():
        for key in ["typicalEmotion", "agentEmotion", "patientEmotion", "negatedEmotion"]:
            emotion_set.add(frame_data[key])

    for emotion_name in sorted(emotion_set):
        emotion_uri = SEO[emotion_name]
        g.add((emotion_uri, RDF.type, emotion_class))
        g.add((emotion_uri, RDFS.label, Literal(emotion_name)))

    # ------------------------------------------------------------------ #
    #  6. Create EmotionFrame instances and LexicalTriggers
    # ------------------------------------------------------------------ #
    for frame_name, frame_data in frames.items():
        frame_uri = SEO[f"frame_{frame_name}"]
        g.add((frame_uri, RDF.type, emotion_frame_class))
        g.add((frame_uri, RDFS.label, Literal(frame_name)))

        # Link to emotions
        g.add((frame_uri, has_typical_emotion, SEO[frame_data["typicalEmotion"]]))
        g.add((frame_uri, has_agent_emotion, SEO[frame_data["agentEmotion"]]))
        g.add((frame_uri, has_patient_emotion, SEO[frame_data["patientEmotion"]]))
        g.add((frame_uri, has_negated_emotion, SEO[frame_data["negatedEmotion"]]))

        # Metadata
        g.add((frame_uri, has_polarity, Literal(frame_data["polarity"], datatype=XSD.string)))
        g.add((frame_uri, has_weight, Literal(frame_data["weight"], datatype=XSD.float)))

        # Description as comment
        if "description" in frame_data:
            g.add((frame_uri, RDFS.comment, Literal(frame_data["description"])))

        # Create LexicalTrigger for each word and link to frame
        for word in frame_data.get("words", []):
            word_id = f"word_{abs(hash(word))}"
            word_uri = SEO[word_id]
            g.add((word_uri, RDF.type, lexical_trigger_class))
            g.add((word_uri, RDFS.label, Literal(word, lang="si")))
            g.add((word_uri, triggers_frame, frame_uri))

    # ------------------------------------------------------------------ #
    #  7. Load and create Modifier instances (for RDF completeness)
    # ------------------------------------------------------------------ #
    try:
        with open(modifiers_path, "r", encoding="utf-8") as f:
            modifiers = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {modifiers_path} not found. Skipping modifiers.")
        modifiers = {}

    # Negation markers
    for word in modifiers.get("negation", {}).get("words", []):
        neg_id = f"neg_{abs(hash(word))}"
        neg_uri = SEO[neg_id]
        g.add((neg_uri, RDF.type, negation_marker_class))
        g.add((neg_uri, RDFS.label, Literal(word, lang="si")))
        g.add((neg_uri, has_effect, Literal("flip_polarity", datatype=XSD.string)))

    # Intensifiers
    for level, level_data in modifiers.get("intensifiers", {}).get("levels", {}).items():
        for word in level_data.get("words", []):
            int_id = f"int_{abs(hash(word))}"
            int_uri = SEO[int_id]
            g.add((int_uri, RDF.type, intensifier_class))
            g.add((int_uri, RDFS.label, Literal(word, lang="si")))
            g.add((int_uri, has_intensity_level, Literal(level, datatype=XSD.string)))

    # Diminishers
    for word in modifiers.get("diminishers", {}).get("words", []):
        dim_id = f"dim_{abs(hash(word))}"
        dim_uri = SEO[dim_id]
        g.add((dim_uri, RDF.type, diminisher_class))
        g.add((dim_uri, RDFS.label, Literal(word, lang="si")))
        g.add((dim_uri, has_effect, Literal("reduce_confidence", datatype=XSD.string)))

    # Discourse connectives
    for conn_type, conn_data in modifiers.get("discourse_connectives", {}).get("types", {}).items():
        for word in conn_data.get("words", []):
            dc_id = f"dc_{abs(hash(word))}"
            dc_uri = SEO[dc_id]
            g.add((dc_uri, RDF.type, discourse_connective_class))
            g.add((dc_uri, RDFS.label, Literal(word, lang="si")))
            g.add((dc_uri, has_effect, Literal(conn_type, datatype=XSD.string)))

    # ------------------------------------------------------------------ #
    #  8. Serialize
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.serialize(destination=output_path, format="turtle")
    print(f"Frame-based ontology generated at {output_path}")

    # ------------------------------------------------------------------ #
    #  9. Verification
    # ------------------------------------------------------------------ #
    # Count entities
    n_frames = len(list(g.subjects(RDF.type, emotion_frame_class)))
    n_triggers = len(list(g.subjects(RDF.type, lexical_trigger_class)))
    n_emotions = len(list(g.subjects(RDF.type, emotion_class)))
    n_neg = len(list(g.subjects(RDF.type, negation_marker_class)))
    n_int = len(list(g.subjects(RDF.type, intensifier_class)))
    n_dim = len(list(g.subjects(RDF.type, diminisher_class)))
    n_dc = len(list(g.subjects(RDF.type, discourse_connective_class)))

    print(f"\nOntology Statistics:")
    print(f"  Emotions:              {n_emotions}")
    print(f"  EmotionFrames:         {n_frames}")
    print(f"  LexicalTriggers:       {n_triggers}")
    print(f"  NegationMarkers:       {n_neg}")
    print(f"  Intensifiers:          {n_int}")
    print(f"  Diminishers:           {n_dim}")
    print(f"  DiscourseConnectives:  {n_dc}")

    # Sample SPARQL verification
    query = """
    SELECT ?word ?frame ?emotion
    WHERE {
        ?w rdf:type seo:LexicalTrigger .
        ?w rdfs:label ?word .
        ?w seo:triggersFrame ?f .
        ?f rdfs:label ?frame .
        ?f seo:hasTypicalEmotion ?e .
        ?e rdfs:label ?emotion .
    } LIMIT 20
    """
    print("\nSample word -> frame -> emotion mappings (first 20):")
    results = g.query(query)
    for row in results:
        print(f"  {row.word} -> [{row.frame}] -> {row.emotion}")


if __name__ == "__main__":
    create_ontology()
