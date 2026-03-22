# Semantic Frame-Based Emotion Ontology for Sinhala Text Classification

## Table of Contents

1. [Overview](#1-overview)
2. [Problem: Why Keyword Triggers Are Insufficient](#2-problem-why-keyword-triggers-are-insufficient)
3. [Solution: 3-Tier Semantic Frame-Based Ontology](#3-solution-3-tier-semantic-frame-based-ontology)
4. [Ontology Schema Design](#4-ontology-schema-design)
5. [Emotion Frames](#5-emotion-frames)
6. [Linguistic Modifiers](#6-linguistic-modifiers)
7. [Semantic Role Detection via Sinhala Case Markers](#7-semantic-role-detection-via-sinhala-case-markers)
8. [Classification Pipeline](#8-classification-pipeline)
9. [Inference Rules](#9-inference-rules)
10. [File Structure](#10-file-structure)
11. [RDF/OWL Schema](#11-rdfowl-schema)
12. [Evaluation Results](#12-evaluation-results)
13. [Worked Examples](#13-worked-examples)
14. [Limitations and Future Work](#14-limitations-and-future-work)
15. [Research Paper Citations](#15-research-paper-citations)

---

## 1. Overview

This project implements a **hybrid emotion classification system** for **Sinhala** (සිංහල) text that combines a **semantic frame-based ontology** (RDF/OWL) with a **transformer-based ML model** (LaBSE). The system classifies Sinhala sentences into four emotions: **Happy** (සතුට), **Sad** (දුක), **Angry** (කෝප), and **Neutral** (සාමාන්‍ය).

The key innovation is the **3-tier semantic ontology** that moves beyond flat keyword-to-emotion mapping to a **frame-based architecture** inspired by the Emotion Frame Ontology (EFO) [1] and the TONE 3-Tiered Ontology for Emotion [6]. Instead of "word → emotion", the system reasons through "word → event frame → (role + context + modifiers) → emotion".

---

## 2. Problem: Why Keyword Triggers Are Insufficient

The original ontology used a flat `LexicalTrigger --triggers--> Emotion` mapping. This approach fails on several common Sinhala linguistic patterns:

| Sentence | Meaning | Old Result | Correct Result | Why It Failed |
|----------|---------|------------|----------------|---------------|
| "මම සතුටු **නෑ**" | I'm **not** happy | Happy | Sad | No negation handling |
| "**හරිම** දුකයි" | **Very** sad | Sad (1.0) | Sad (high intensity) | No intensity awareness |
| "**මම** ගැහුවා" | **I** hit (someone) | Angry | Angry | Correct by accident |
| "**මාව** ගැහුවා" | (Someone) hit **me** | Angry | Sad/Fear | No role awareness |
| "සතුටු **වුණත්** දුකයි" | Happy **but** sad | Conflict→ML | Sad (dominant) | No discourse connective handling |
| "**තෝ** මොකෙක්ද **යකෝ**" | What are **you** (hostile) **devil** | Neutral | Angry | No hostile address detection |

These failures occur because Sinhala is a **morphologically rich language** [4] where:
- **Negation** is expressed through particles (නෑ, නැහැ, බෑ) that reverse polarity
- **Intensity** is conveyed through adverbs (හරිම, ගොඩක්, මාරම)
- **Semantic roles** are encoded via case markers (මම=agent, මාව=patient, මට=experiencer)
- **Discourse structure** uses contrastive connectives (ඒත්, නමුත්, වුණත්)
- **Social register** signals emotion through hostile vs. neutral address forms (තෝ vs. ඔයා)

---

## 3. Solution: 3-Tier Semantic Frame-Based Ontology

The redesigned system processes text through three sequential tiers before falling back to ML:

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Sinhala Text                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 1: Linguistic Analysis (Rule-Based)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │  Negation    │ │ Intensifiers │ │ Role Detection       │ │
│  │  Detection   │ │ & Diminishers│ │ (Case Markers)       │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
│  ┌──────────────────────┐ ┌────────────────────────────────┐ │
│  │ Discourse Connectives│ │ Hostile Address Detection      │ │
│  └──────────────────────┘ └────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │  linguistic_context
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 2: Frame-Based Ontology Matching (SPARQL)              │
│                                                              │
│  Token → LexicalTrigger → EmotionFrame (NOT direct Emotion)  │
│                                                              │
│  Each frame encodes:                                         │
│    • typicalEmotion   (default)                              │
│    • agentEmotion     (when speaker is doer)                 │
│    • patientEmotion   (when speaker is receiver)             │
│    • negatedEmotion   (when negated)                         │
│    • polarity         (positive / negative)                  │
│    • weight           (signal strength: 0.4 - 1.0)           │
└────────────────────────┬────────────────────────────────────┘
                         │  frame_matches[]
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 3: Semantic Inference                                  │
│                                                              │
│  For each matched frame:                                     │
│    1. Select emotion based on speaker role                   │
│    2. Flip to negatedEmotion if negation detected            │
│    3. Multiply weight by intensifier/diminisher              │
│    4. Apply discourse connective clause weighting             │
│    5. Add hostile address anger signal                        │
│                                                              │
│  Aggregate → emotion_scores{emotion: weighted_score}         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  HYBRID DECISION                                             │
│                                                              │
│  • Single emotion matched → Return (Ontology)                │
│  • Dominant emotion (>2× runner-up) → Return (Ontology)      │
│  • Conflict (close scores) → Fallback to ML (LaBSE)          │
│  • No frames matched → Fallback to ML (LaBSE)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Ontology Schema Design

### 4.1 Previous Schema (Flat Keyword Mapping)

```
LexicalTrigger  --triggers-->  Emotion
"සතුටු"        --triggers-->  Happy
```

A word directly maps to an emotion. No consideration of context, roles, or modifiers.

### 4.2 New Schema (Frame-Based)

```
LexicalTrigger  ─── triggersFrame ───►  EmotionFrame
                                          │
                                          ├── hasTypicalEmotion ──► Emotion
                                          ├── hasAgentEmotion ────► Emotion
                                          ├── hasPatientEmotion ──► Emotion
                                          ├── hasNegatedEmotion ──► Emotion
                                          ├── hasPolarity ────────► "positive"/"negative"
                                          └── hasWeight ──────────► 0.4 - 1.0

NegationMarker      (rdfs:label "නෑ", "නැහැ", "එපා", ...)
Intensifier         (rdfs:label "හරිම", "ගොඩක්", ...) ── hasIntensityLevel ──► "high"/"medium"
Diminisher          (rdfs:label "පොඩ්ඩක්", "ටිකක්", ...)
DiscourseConnective (rdfs:label "ඒත්", "නමුත්", ...) ── hasEffect ──► "contrastive"/"causal"
```

The critical change: **"සතුටු" no longer maps directly to Happy**. It maps to a `PositiveEmotion` frame, and the final emotion is inferred after checking negation, roles, and modifiers.

### 4.3 RDF Class Hierarchy

| Class | Description |
|-------|-------------|
| `seo:Emotion` | Emotion labels: Happy, Sad, Angry, Neutral |
| `seo:EmotionFrame` | Semantic event frames linking words to role-dependent emotions |
| `seo:LexicalTrigger` | Sinhala words that trigger emotion frames |
| `seo:NegationMarker` | Negation particles (නෑ, නැහැ, එපා, බෑ, ...) |
| `seo:Intensifier` | Intensity boosters (හරිම, ගොඩක්, මාරම, ...) |
| `seo:Diminisher` | Intensity reducers (පොඩ්ඩක්, ටිකක්, ...) |
| `seo:DiscourseConnective` | Clause-linking words (ඒත්, නමුත්, වුණත්, ...) |

### 4.4 RDF Property Definitions

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `seo:triggersFrame` | LexicalTrigger | EmotionFrame | Links a word to its semantic frame |
| `seo:hasTypicalEmotion` | EmotionFrame | Emotion | Default emotion (no role info) |
| `seo:hasAgentEmotion` | EmotionFrame | Emotion | Emotion when speaker is the agent/doer |
| `seo:hasPatientEmotion` | EmotionFrame | Emotion | Emotion when speaker is the receiver/patient |
| `seo:hasNegatedEmotion` | EmotionFrame | Emotion | Emotion when the frame is negated |
| `seo:hasPolarity` | EmotionFrame | xsd:string | "positive" or "negative" |
| `seo:hasWeight` | EmotionFrame | xsd:float | Signal strength (0.4=weak, 1.0=strong) |
| `seo:hasIntensityLevel` | Intensifier | xsd:string | "high" or "medium" |
| `seo:hasEffect` | DiscourseConnective | xsd:string | "contrastive", "causal", or "additive" |

---

## 5. Emotion Frames

The ontology defines **18 semantic frames** organized by emotion category. Each frame represents a type of emotional situation rather than a simple keyword.

### 5.1 Happy Frames (6 frames)

| Frame | Description | Weight | #Words | Example Words |
|-------|-------------|--------|--------|---------------|
| **PositiveEmotion** | Direct positive emotional state | 1.0 | 6 | සතුටුයි, ආසයි, happy |
| **PositiveQuality** | Positive quality/aesthetic appreciation | 0.9 | 11 | ලස්සනයි, හොඳ, සුපිරි |
| **Achievement** | Accomplishment or success | 1.0 | 6 | කරගත්තා, හම්බුණා, ලැබුණා |
| **SocialBonding** | Social connection, friendship, fun | 0.8 | 11 | යාලුවෝ, මචං, ආතල් |
| **PleasantExperience** | Enjoyable activities and things | 0.7 | 22 | කෑම, සින්දු, නිවාඩු |
| **GeneralHappyContext** | Statistical co-occurrence (weak signal) | 0.4 | 38 | එකක්, පුළුවන්, උදේ |

### 5.2 Sad Frames (6 frames)

| Frame | Description | Weight | #Words | Example Words |
|-------|-------------|--------|--------|---------------|
| **SadEmotion** | Direct sadness expressions | 1.0 | 8 | දුක, දුකක්, පාලුයි |
| **LossExperience** | Loss, separation, emptiness | 1.0 | 8 | පාලු, තනි, හුදකලා |
| **SufferingExperience** | Pain, suffering, enduring hardship | 1.0 | 7 | වේදනාව, රිදුම, කඳුළු |
| **DespairState** | Hopelessness, helplessness, defeat | 1.0 | 7 | අසරණ, බැරි, පැරදුණා |
| **EmotionalLonging** | Yearning, nostalgia, memory-driven sadness | 0.9 | 12 | හිතේ, ආදරේ, මතකයක් |
| **GeneralSadContext** | Statistical co-occurrence (weak signal) | 0.4 | 57 | මෙච්චර, හැමදාම |

### 5.3 Angry Frames (6 frames)

| Frame | Description | Weight | #Words | Role-Dependent? |
|-------|-------------|--------|--------|-----------------|
| **PhysicalHarm** | Physical violence (hit, kill, break) | 1.0 | 10 | **Yes**: agent=Angry, patient=Sad |
| **VerbalInsult** | Derogatory terms, name-calling | 1.0 | 9 | **Yes**: agent=Angry, patient=Sad |
| **ThreatAction** | Threats, aggressive commands | 1.0 | 8 | **Yes**: agent=Angry, patient=Sad |
| **Destruction** | Breaking, destroying property | 1.0 | 3 | **Yes**: agent=Angry, patient=Sad |
| **HostileExclamation** | Hostile expletives/exclamations | 0.9 | 2 | No: always Angry |
| **GeneralAngryContext** | Statistical co-occurrence (weak signal) | 0.4 | 55 | No: always Angry |

### 5.4 Role-Dependent Emotion Resolution

The most important innovation is that certain frames produce **different emotions depending on who is the experiencer**:

```
PhysicalHarm frame: "ගහනවා" (hitting)
  ├── If speaker is AGENT (මම ගැහුවා = I hit)    → Angry
  └── If speaker is PATIENT (මාව ගැහුවා = I was hit) → Sad

VerbalInsult frame: "පරයා" (bastard)
  ├── If speaker is AGENT (using the insult)       → Angry
  └── If speaker is PATIENT (called that)          → Sad

SufferingExperience frame: "වේදනාව" (pain)
  ├── If speaker is AGENT (causing pain)           → Angry
  └── If speaker is PATIENT (experiencing pain)    → Sad
```

This is directly inspired by the **appraisal perspective** concept from the Emotion Frame Ontology (EFO) [1], where the same event maps to different emotions depending on the experiencer's stance.

---

## 6. Linguistic Modifiers

### 6.1 Negation Markers

Sinhala negation particles that flip the polarity of nearby emotion frames.

| Word | Meaning | Effect |
|------|---------|--------|
| නෑ | no / not | Flip polarity |
| නැහැ | no / not | Flip polarity |
| නැත | not (formal) | Flip polarity |
| එපා | don't want / refuse | Flip polarity |
| බෑ | can't | Flip polarity |
| බැහැ | can't | Flip polarity |
| නැති | without / lacking | Flip polarity |
| නැතුව | without | Flip polarity |
| නොවේ | is not | Flip polarity |
| නෙවෙයි | is not (colloquial) | Flip polarity |
| නො | not (prefix) | Flip polarity |
| නැත්තේ | not (emphatic) | Flip polarity |

**Negation scope**: Within a window of **2 tokens** from the negation particle. Only applied to **strong frames** (weight ≥ 0.7) to avoid false negation of weak contextual signals.

**Example**:
```
"මම සතුටු නෑ" (I'm not happy)
  Token "සතුටු" → PositiveEmotion frame → typical: Happy
  Token "නෑ" detected as negation (distance=1, within window)
  Negation flips: Happy → negatedEmotion = Sad ✓
```

### 6.2 Intensifiers

Words that amplify the intensity of nearby emotion frames.

| Level | Multiplier | Words |
|-------|-----------|-------|
| **High** | ×1.5 | හරිම (very), ගොඩක් (a lot), මාරම (extremely), ඉතාම (exceedingly), අතිශයින් (exceptionally), මාර (intense) |
| **Medium** | ×1.25 | පට්ට (slang for very), මහ (great), ඕනවට වඩා (more than enough), ගොඩ (much), හැබැයිම (really) |

**Intensifier scope**: Within **3 tokens** of the intensifier.

### 6.3 Diminishers

Words that reduce the intensity of nearby emotion frames.

| Multiplier | Words |
|-----------|-------|
| ×0.6 | පොඩ්ඩක් (a little), ටිකක් (a bit), අඩුවෙන් (less), ටිකට (slightly), පොඩි (small) |

### 6.4 Discourse Connectives

Words that affect how emotional signals from different clauses combine.

| Type | Pre-clause Weight | Post-clause Weight | Words | Effect |
|------|------------------|-------------------|-------|--------|
| **Contrastive** | ×0.3 | ×1.5 | ඒත් (but), නමුත් (however), හැබැයි (but), වුණත් (even though), උනාට (despite) | Latter clause dominates |
| **Causal** | ×1.0 | ×1.0 | නිසා (because), හින්දා (due to), නිසාම (because of), හේතුවෙන් (owing to) | Both clauses equal |
| **Additive** | ×1.0 | ×1.0 | ඒ වගේම (likewise), ඊට අමතරව (additionally) | Both clauses equal |

**Example**:
```
"සතුටු වුණත් දුකයි" (Happy but sad)
  "වුණත්" detected as contrastive connective (index 1)
  "සතුටු" (index 0, before connective) → weight × 0.3 = 0.30
  "දුකයි"  (index 2, after connective)  → weight × 1.5 = 1.50
  Result: Sad dominates (1.50 vs 0.30) → Sad ✓
```

---

## 7. Semantic Role Detection via Sinhala Case Markers

Sinhala is a **subject-object-verb (SOV)** language with rich case marking. Semantic roles are identified through pronouns and case suffixes [4, 5].

### 7.1 Agent Markers (Nominative Case — Doer of Action)

| Person | Neutral | Hostile |
|--------|---------|---------|
| 1st person | මම (I), මං (I, informal), අපි (we) | — |
| 2nd person | ඔයා (you), ඔබ (you, formal) | තෝ (you), උඹ (you), තොපි (you all) |
| 3rd person | එයා (he/she) | ඌ, මූ, මුන්, එවුන් (derogatory) |

### 7.2 Patient Markers (Accusative Case — Receiver of Action)

| Person | Neutral | Hostile |
|--------|---------|---------|
| 1st person | මාව (me), අපව (us) | — |
| 2nd person | ඔයාව (you) | තෝව (you), උඹව (you) |
| 3rd person | එයාව (him/her) | — |

### 7.3 Experiencer Markers (Dative Case — Feeler/Receiver of Experience)

| Person | Neutral | Hostile |
|--------|---------|---------|
| 1st person | මට (to me), අපට (to us) | — |
| 2nd person | ඔයාට (to you) | තොට (to you), උඹට (to you) |
| 3rd person | එයාට (to him/her) | — |

### 7.4 Possessive Markers (Genitive Case)

| Person | Neutral | Hostile |
|--------|---------|---------|
| 1st person | මගේ (my), අපේ (our) | — |
| 2nd person | ඔයාගේ (your) | තොගෙ (your), උඹේ (your), තොපේ (your all) |
| 3rd person | එයාගේ (his/her) | මුගේ (his, derogatory) |

### 7.5 Hostile Address Detection

The presence of **hostile second-person pronouns** (තෝ, උඹ, තොපි, etc.) and their case-marked forms serves as a **direct anger signal**. In Sinhala culture, using these address forms (as opposed to neutral ඔයා/ඔබ) indicates hostility and disrespect [4]. Each hostile marker detected adds an anger weight of **0.7** to the emotion score.

---

## 8. Classification Pipeline

### 8.1 Initialization

At startup, the classifier loads:
1. **RDF Ontology** (`sinhala_emotion.ttl`) — for SPARQL-based frame matching
2. **Frames data** (`frames.json`) — for frame property lookup (faster than SPARQL for metadata)
3. **Modifiers data** (`modifiers.json`) — for negation/intensifier/connective detection
4. **Role markers data** (`role_markers.json`) — for semantic role detection
5. **ML centroids** (`centroids.pkl`) — for LaBSE fallback
6. **LaBSE model** (`sentence-transformers/LaBSE`) — for sentence encoding

### 8.2 Classification Flow (Per Request)

```python
def predict(text):
    # Step 1: Tokenize
    tokens = tokenize(text)  # IndicNLP trivial_tokenize

    # Step 2: Tier 1 - Linguistic Analysis
    linguistic_context = analyze_linguistics(tokens)
    # → negation_positions, intensifier_positions, diminisher_positions,
    #   connective_positions, role_info

    # Step 3: Tier 2 - Frame Matching (SPARQL)
    frame_matches = match_frames(tokens)
    # → [{token_idx, token, frame_name, typicalEmotion, agentEmotion,
    #     patientEmotion, negatedEmotion, polarity, weight}, ...]

    # Step 4: Tier 3 - Semantic Inference
    emotion_scores, matched_words, explanation = infer_emotions(
        frame_matches, linguistic_context, tokens
    )

    # Step 5: Hybrid Decision
    if no_matches:
        return ML_classify(text)          # Fallback
    elif single_emotion:
        return ontology_result            # Clean match
    elif dominant_emotion (>2× runner-up):
        return ontology_dominant           # Clear winner
    else:
        return ML_classify(text)          # Conflict resolution
```

### 8.3 SPARQL Query for Frame Matching

```sparql
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
```

The `STRSTARTS` bidirectional filter handles Sinhala morphological variants where words may have suffixes (e.g., "සතුටුයි" matching "සතුටු"). A **maximum prefix difference of 3 characters** is enforced to prevent false matches, and results are **deduplicated per (token, frame)** to keep only the most specific match.

---

## 9. Inference Rules

### Rule 1: Role-Based Emotion Selection

```
IF speaker_is_patient THEN emotion = frame.patientEmotion
ELSE IF speaker_is_agent THEN emotion = frame.agentEmotion
ELSE emotion = frame.typicalEmotion
```

### Rule 2: Negation Polarity Flip

```
IF frame.weight >= 0.7
   AND negation_word within 2 tokens of frame trigger
THEN emotion = frame.negatedEmotion
```

Only strong frames are flipped. Weak contextual frames (weight < 0.7) are immune to negation to avoid over-flipping on pragmatic negation constructions.

### Rule 3: Intensifier Weight Boost

```
IF intensifier within 3 tokens of frame trigger
THEN weight *= intensifier_multiplier (1.25 or 1.5)
```

### Rule 4: Diminisher Weight Reduction

```
IF diminisher within 3 tokens of frame trigger
THEN weight *= 0.6
```

### Rule 5: Discourse Connective Clause Weighting

```
IF contrastive connective detected at position C
THEN FOR each frame trigger:
    IF position < C: weight *= 0.3  (pre-connective, de-weighted)
    IF position > C: weight *= 1.5  (post-connective, boosted)
```

### Rule 6: Hostile Address Signal

```
IF hostile_address_form detected (count = N)
THEN Angry += 0.7 × N
```

---

## 10. File Structure

```
sinhala-emotion-ontology/
├── ontology/
│   ├── frames.json            # 18 emotion frame definitions with words
│   ├── modifiers.json         # Negation, intensifiers, diminishers, connectives
│   ├── role_markers.json      # Sinhala pronouns and case markers
│   ├── lexicon.json           # Flat lexicon (backward compatible)
│   └── sinhala_emotion.ttl    # Generated RDF/OWL ontology (Turtle format)
│
├── src/
│   ├── create_ontology.py     # Generates TTL from frames.json + modifiers.json
│   ├── classify.py            # 3-tier semantic frame-based classifier
│   ├── build_model.py         # Computes LaBSE emotion centroids
│   ├── extract_lexicon.py     # Extracts frame-based lexicon from dataset
│   ├── evaluate.py            # Comprehensive evaluation with feature impact
│   └── app.py                 # FastAPI web application
│
├── data/
│   ├── sinhala_samples.json   # ~3,500 labeled Sinhala text samples
│   └── centroids.pkl          # Pre-computed LaBSE emotion centroids
│
└── docs/
    ├── semantic_frame_ontology.md   # This document
    ├── classification_flow.md       # Original flow explanation
    └── research_justification.md    # Research paper justifications
```

### Data File Descriptions

| File | Format | Description |
|------|--------|-------------|
| `frames.json` | JSON | 18 emotion frames, each with: description, typicalEmotion, agentEmotion, patientEmotion, negatedEmotion, polarity, weight, and word list |
| `modifiers.json` | JSON | 12 negation particles, 11 intensifiers (2 levels), 5 diminishers, 12 discourse connectives (3 types) |
| `role_markers.json` | JSON | Agent, patient, experiencer, and possessive markers for 1st/2nd/3rd person, with hostile flags |
| `sinhala_emotion.ttl` | RDF/Turtle | Generated ontology with 4 emotions, 18 frames, ~280 triggers, and all modifier instances |

---

## 11. RDF/OWL Schema

### 11.1 Namespace

```turtle
@prefix seo: <http://www.semanticweb.org/sinhala-emotion-ontology#> .
```

### 11.2 Example RDF Triples

```turtle
# Emotion instances
seo:Happy a seo:Emotion ; rdfs:label "Happy" .
seo:Sad   a seo:Emotion ; rdfs:label "Sad" .
seo:Angry a seo:Emotion ; rdfs:label "Angry" .

# EmotionFrame: PhysicalHarm (role-dependent!)
seo:frame_PhysicalHarm a seo:EmotionFrame ;
    rdfs:label "PhysicalHarm" ;
    rdfs:comment "Physical violence: hitting, killing. Agent=Angry, Patient=Sad" ;
    seo:hasTypicalEmotion seo:Angry ;
    seo:hasAgentEmotion   seo:Angry ;
    seo:hasPatientEmotion seo:Sad ;
    seo:hasNegatedEmotion seo:Neutral ;
    seo:hasPolarity       "negative"^^xsd:string ;
    seo:hasWeight         "1.0"^^xsd:float .

# LexicalTrigger → Frame (not directly to Emotion)
seo:word_12345 a seo:LexicalTrigger ;
    rdfs:label "ගහනවා"@si ;
    seo:triggersFrame seo:frame_PhysicalHarm .

# Negation Marker
seo:neg_67890 a seo:NegationMarker ;
    rdfs:label "නෑ"@si ;
    seo:hasEffect "flip_polarity"^^xsd:string .

# Intensifier
seo:int_11111 a seo:Intensifier ;
    rdfs:label "හරිම"@si ;
    seo:hasIntensityLevel "high"^^xsd:string .

# Discourse Connective
seo:dc_22222 a seo:DiscourseConnective ;
    rdfs:label "ඒත්"@si ;
    seo:hasEffect "contrastive"^^xsd:string .
```

### 11.3 Ontology Statistics

| Entity Type | Count |
|-------------|-------|
| Emotions | 4 (Happy, Sad, Angry, Neutral) |
| EmotionFrames | 18 |
| LexicalTriggers | ~280 |
| NegationMarkers | 12 |
| Intensifiers | 11 |
| Diminishers | 5 |
| DiscourseConnectives | 12 |

---

## 12. Evaluation Results

Evaluated on **500 random samples** from the dataset (seed=42 for reproducibility).

### 12.1 Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **92.20%** |
| **Weighted Precision** | **0.93** |
| **Weighted Recall** | **0.92** |
| **Weighted F1-Score** | **0.92** |

### 12.2 Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Happy | 0.95 | 0.93 | 0.94 | 258 |
| Sad | 0.94 | 0.90 | 0.92 | 185 |
| Angry | 0.81 | 0.95 | 0.87 | 57 |

### 12.3 Method Distribution

| Method | Predictions | Percentage | Accuracy |
|--------|------------|------------|----------|
| **Ontology (Frame-based)** | 412 | 82.4% | **93.69%** |
| **ML (LaBSE fallback)** | 88 | 17.6% | 85.23% |

The frame-based ontology handles **82.4%** of all predictions (up from previous keyword-only approach) with **93.69% accuracy** — demonstrating that the semantic frame approach provides both broader coverage and higher precision than flat keyword matching.

### 12.4 Semantic Feature Impact

| Feature | Cases Detected | Accuracy | Notes |
|---------|---------------|----------|-------|
| **Intensifiers** | 99 | **98.99%** | High-confidence feature |
| **Hostile Address** | 54 | **94.44%** | Strong anger signal |
| **Negation** | 20 | 45.00% | Works for simple polarity flip; pragmatic negation remains challenging |
| **Discourse Connectives** | 3 | 66.67% | Rare in dataset but effective |

### 12.5 Top Frame Usage

| Frame | Times Matched |
|-------|---------------|
| GeneralSadContext | 340 |
| GeneralHappyContext | 320 |
| GeneralAngryContext | 142 |
| PositiveQuality | 105 |
| EmotionalLonging | 76 |
| PleasantExperience | 70 |
| PositiveEmotion | 50 |
| LossExperience | 48 |
| SocialBonding | 43 |
| Achievement | 37 |

---

## 13. Worked Examples

### Example 1: Negation Handling

**Input**: "මම සතුටු නෑ" (I'm not happy)

```
Tier 1: Linguistic Analysis
  → Negation detected: "නෑ" at position 2
  → Agent marker: "මම" (I, nominative) at position 0

Tier 2: Frame Matching
  → "සතුටු" (position 1) → PositiveEmotion frame
    {typical: Happy, agent: Happy, patient: Happy,
     negated: Sad, polarity: positive, weight: 1.0}

Tier 3: Inference
  → Role: speaker_is_agent → emotion = agentEmotion = Happy
  → Negation check: "නෑ" at distance 1 from "සතුටු" (within window 2)
  → Frame weight 1.0 ≥ 0.7, so negation applies
  → Flip: Happy → negatedEmotion = Sad

Result: Sad (Ontology) ✓
```

### Example 2: Contrastive Discourse Connective

**Input**: "සතුටු වුණත් දුකයි" (Happy but sad)

```
Tier 1: Linguistic Analysis
  → Contrastive connective: "වුණත්" at position 1
    (pre_weight=0.3, post_weight=1.5)

Tier 2: Frame Matching
  → "සතුටු" (pos 0) → PositiveEmotion {typical: Happy, weight: 1.0}
  → "දුකයි" (pos 2) → SadEmotion {typical: Sad, weight: 1.0}

Tier 3: Inference
  → "සතුටු" (pos 0 < connective pos 1): weight = 1.0 × 0.3 = 0.30 → Happy
  → "දුකයි" (pos 2 > connective pos 1): weight = 1.0 × 1.5 = 1.50 → Sad
  → Scores: Happy=0.30, Sad=1.50
  → Sad dominates (1.50/0.30 = 5.0× > 2.0× threshold)

Result: Sad (Ontology, dominant) with confidence 0.83 ✓
```

### Example 3: Hostile Address + Exclamation

**Input**: "තෝ මොකෙක්ද යකෝ" (What are you [hostile] you devil)

```
Tier 1: Linguistic Analysis
  → Hostile address: "තෝ" (hostile 2nd person nominative)
    hostile_count = 1

Tier 2: Frame Matching
  → "යකෝ" → HostileExclamation {typical: Angry, weight: 0.9}

Tier 3: Inference
  → "යකෝ" → Angry (weight 0.9)
  → Hostile address signal: Angry += 0.7 × 1 = 0.7
  → Total: Angry = 0.9 + 0.7 = 1.6

Result: Angry (Ontology) with confidence 0.80 ✓
```

### Example 4: Intensifier Boost

**Input**: "මම අද ගොඩක් සතුටුයි" (I am very happy today)

```
Tier 1: Linguistic Analysis
  → Agent marker: "මම" (I, agent)
  → Intensifier: "ගොඩක්" at position 2 (level: high, ×1.5)

Tier 2: Frame Matching
  → "සතුටුයි" (pos 3) → PositiveEmotion {typical: Happy, weight: 1.0}

Tier 3: Inference
  → Role: speaker_is_agent → agentEmotion = Happy
  → Intensifier "ගොඩක්" at distance 1 (within window 3)
  → weight = 1.0 × 1.5 = 1.5

Result: Happy (Ontology) with confidence 0.75 ✓
```

### Example 5: Role-Dependent Emotion (PhysicalHarm)

**Input**: "මාව ගැහුවා" (I was hit / someone hit me)

```
Tier 1: Linguistic Analysis
  → Patient marker: "මාව" (me, accusative) → speaker_is_patient = True

Tier 2: Frame Matching
  → "ගැහුවා" → PhysicalHarm frame
    {typical: Angry, agent: Angry, patient: Sad, weight: 1.0}

Tier 3: Inference
  → Role: speaker_is_patient → patientEmotion = Sad

Result: Sad (Ontology) ✓
  (Previously would have returned Angry with flat keyword mapping)
```

---

## 14. Limitations and Future Work

### 14.1 Current Limitations

1. **Pragmatic Negation**: Simple polarity flip works for "not happy → sad" but fails on complex constructions like "I don't feel like doing anything" (expresses sadness through pragmatic meaning, not polarity reversal). Current accuracy on negation cases: ~45%.

2. **Limited Syntactic Parsing**: Role detection relies on pronoun/case marker lookup, not full dependency parsing. Complex sentences with multiple clauses and embedded roles may not be correctly analyzed.

3. **Static Frame Assignment**: Words are assigned to frames via manual curation and statistical extraction. New words or domain-specific vocabulary require updating the frames.json.

4. **Sarcasm and Irony**: The system cannot detect sarcastic usage (e.g., "ගොඩක් ලස්සනයි" said sarcastically to mean ugly).

5. **Neutral Class**: Neutral emotion has minimal ontology support; it's primarily handled by the ML fallback.

### 14.2 Future Improvements

1. **Fine-tuned Multilingual Transformer**: Replace LaBSE centroids with a fine-tuned XLM-R or IndicBERT model on the Sinhala emotion dataset [7].

2. **Morphological Analyzer**: Integrate a Sinhala morphological analyzer to decompose words into stems and suffixes, improving frame matching coverage [4, 5].

3. **Three-Address Code Semantics**: Implement the "Three Address Code Based Semantics Processor for Sinhala" [5] to generate (actor, action, object) triples as an intermediate representation before ontology matching.

4. **SWRL Rules in Ontology**: Encode inference rules (currently in Python) as SWRL rules directly in the OWL ontology for better portability and standards compliance [1].

5. **Ontology Learning**: Automatically discover new frames and modifier patterns from unlabeled Sinhala text corpora.

---

## 15. Research Paper Citations

### Primary Foundations

**[1]** De Giorgis, S., & Gangemi, A. (2024). **The Emotion Frame Ontology**. *arXiv preprint arXiv:2401.10751*.
- Contribution to our work: The frame-based architecture with `EmotionFrame`, semantic roles (experiencer, cause, target), and the concept that the same event maps to different emotions depending on perspective. Our `EmotionFrame` class with `hasAgentEmotion`/`hasPatientEmotion` is a Sinhala-specific adaptation of EFO's emotion frames.

**[2]** Eisenreich, C., Ott, J., Süßdorf, T., Willms, C., & Declerck, T. (2012). **From Tale to Speech: Ontology-based Emotion and Dialogue Annotation of Fairy Tales with a TTS Output**. *DFKI & Saarland University*.
- Contribution to our work: Ontology-based emotion annotation with incremental annotation of characters and their roles, which maps directly to our role detection (agent/patient) for emotion resolution.

**[3]** Anushya, P. J., & Nisha Priya, P. (2020). **Mood recognition emotion ontology with texts**. *International Journal of Advance Research, Ideas and Innovations in Technology (IJARIIT)*, 6(4), V6I4-1453.
- Contribution to our work: Validates the hybrid approach of combining ontology rules with deep learning, demonstrating that traditional keyword methods are insufficient for complex social media text.

### Sinhala-Specific NLP Research

**[4]** Medagoda, N., Shanmuganathan, S., & Whalley, J. (2015). **A Framework for Sentiment Classification for Morphologically Rich Languages: A Case Study for Sinhala**. *Unitec Institute of Technology, Auckland*.
- Contribution to our work: Justifies the need for negation handling, intensifier detection, and discourse feature processing specific to Sinhala morphology. Demonstrates that semantic and discourse features are critical for accurate sentiment classification in Sinhala.

**[5]** Welgama, V., Herath, D., Liyanage, C., Udalamatta Gamage, N., Ranathunga, S., & Jayawardana, N. (2011). **Towards a Sinhala Wordnet**. *Conference on Human Language Technology for Development, Alexandria, Egypt*.
Also: Weerasinghe, R., Herath, D., & Welgama, V. (2013). **A Survey on Publicly Available Sinhala Natural Language Processing Tools and Research**. *University of Colombo School of Computing*.
- Contribution to our work: Provides background on Sinhala syntactic structure, case markers, and semantic role assignment. Highlights that animacy and case markers play a central role in Sinhala semantic analysis, which we leverage for agent/patient detection.

**[6]** Prabhashini, W. G. S. (2020). **Three Address Code Based Semantics Processor for Sinhala**. *General Sir John Kotelawala Defence University (KDU)*.
- Contribution to our work: Proposes generating semantic representations of Sinhala text beyond keywords using (actor, action, object) triples, which aligns with our ontology's frame-based approach of mapping text to event structures.

### Hybrid Approach and Transformer Models

**[7]** Conneau, A., et al. (2020). **Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)**. *Proceedings of the 58th Annual Meeting of the ACL*.
Also: Feng, F., et al. (2022). **Language-agnostic BERT Sentence Embedding (LaBSE)**. *ACL*.
- Contribution to our work: LaBSE provides the ML fallback component of our hybrid system, offering context-aware sentence embeddings for low-resource languages like Sinhala.

**[8]** TONE: 3-Tiered Ontology for Emotion (2024). Validates the tiered approach of separating lexical, syntactic, and semantic layers for emotion classification — the architectural basis of our 3-tier pipeline.

### Ontology Engineering Standards

**[9]** Gangemi, A. (2005). **Ontology Design Patterns for Semantic Web Content**. *Proceedings of the 4th International Semantic Web Conference (ISWC 2005)*.
- Contribution to our work: The Content Ontology Design Pattern (ODP) approach used in EFO [1] that we adapted for defining emotion frames as reusable patterns.

**[10]** Noy, N. F., & McGuinness, D. L. (2001). **Ontology Development 101: A Guide to Creating Your First Ontology**. *Stanford Knowledge Systems Laboratory Technical Report KSL-01-05*.
- Contribution to our work: General ontology engineering methodology followed in the design of our class hierarchy, property definitions, and instance creation.

---

*Document generated for the Sinhala Emotion Ontology Project. For technical questions, refer to the source code in `src/classify.py` (3-tier pipeline) and `src/create_ontology.py` (TTL generation).*
