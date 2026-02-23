"""
extract_lexicon.py -- Extract frame-based lexicon from Sinhala emotion dataset.

Outputs:
  1. ontology/lexicon.json         -- flat lexicon (backward compatible)
  2. ontology/frames.json          -- frame-based lexicon (new format)
  3. ontology/modifiers.json       -- modifiers (updated with patterns from data)
  4. ontology/role_markers.json    -- role markers (unchanged, curated manually)

The script:
  - Tokenizes all samples and counts word frequencies per emotion.
  - Identifies exclusive words (appear in only one emotion category).
  - Distributes words into semantic frames based on heuristic rules.
  - Also reports negation, intensifier, and connective usage statistics.
"""

import json
import os
from collections import Counter
from indicnlp.tokenize import indic_tokenize

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(base_dir, "data", "sinhala_samples.json")
lexicon_file = os.path.join(base_dir, "ontology", "lexicon.json")
frames_file = os.path.join(base_dir, "ontology", "frames.json")
modifiers_file = os.path.join(base_dir, "ontology", "modifiers.json")

# Parameters
TOP_N_CANDIDATES = 300  # Initial candidates to consider
FINAL_N = 100           # Final number of words per emotion
MIN_LENGTH = 3          # Minimum token length

# -----------------------------------------------------------------
# Known modifier words (to exclude from frame assignment)
# -----------------------------------------------------------------
KNOWN_NEGATION = {"නෑ", "නැහැ", "නැත", "එපා", "බෑ", "බැහැ", "නැති",
                  "නැතුව", "නොවේ", "නෙවෙයි", "නො", "නැත්තේ", "නැද්ද", "නෑනේ"}
KNOWN_INTENSIFIERS = {"හරිම", "ගොඩක්", "මාරම", "ඉතාම", "අතිශයින්", "මාර",
                      "පට්ට", "මහ", "ඕනවට වඩා", "ගොඩ", "හැබැයිම"}
KNOWN_DIMINISHERS = {"පොඩ්ඩක්", "ටිකක්", "අඩුවෙන්", "ටිකට", "පොඩි"}
KNOWN_CONNECTIVES = {"ඒත්", "නමුත්", "හැබැයි", "වුණත්", "උනාට",
                     "නිසා", "හින්දා", "නිසාම", "හේතුවෙන්"}
ALL_MODIFIERS = KNOWN_NEGATION | KNOWN_INTENSIFIERS | KNOWN_DIMINISHERS | KNOWN_CONNECTIVES

# -----------------------------------------------------------------
# Known role markers (to exclude from frame assignment)
# -----------------------------------------------------------------
KNOWN_ROLE_MARKERS = {
    "මම", "මං", "අපි", "ඔයා", "ඔබ", "ඔබලා",
    "තෝ", "උඹ", "තොපි",
    "එයා", "ඌ", "මූ", "මුන්", "එවුන්",
    "මාව", "මව", "අපව", "ඔයාව", "තෝව", "උඹව", "එයාව",
    "මට", "අපට", "ඔයාට", "ඔබට", "තොට", "උඹට", "එයාට", "ඌට", "එවුන්ට",
    "මගේ", "මගෙ", "මගේම", "අපේ", "ඔයාගේ", "ඔබගේ",
    "තොගෙ", "උඹේ", "තොපේ", "මුගේ", "එයාගේ"
}

# -----------------------------------------------------------------
# Heuristic frame assignment rules
# -----------------------------------------------------------------
# These sets define "seed" words for each frame.
# Words not matching any seed set go to General*Context frames.

FRAME_SEEDS = {
    # Happy frames
    "PositiveEmotion": {"සතුටුයි", "සතුටු", "ආසයි", "ආසම", "happy", "හිනාව", "සතුට", "සතුටින්"},
    "PositiveQuality": {"ලස්සනයි", "ලස්සන", "හොඳ", "සුපිරි", "රහයි", "මරු", "ෆේවරිට්",
                        "වටිනවා", "අලුත්", "හොඳට", "ෆුල්"},
    "Achievement": {"කරගත්තා", "හම්බුණා", "ලැබුණා", "ලැබුණ", "කරා", "දුන්නා",
                    "දිනුවා", "පාස්", "ජය", "ජයග්‍රහණය"},
    "SocialBonding": {"යාලුවෝ", "යාලුවෙක්", "මචං", "පාටියක්", "ගැම්මක්", "ෆන්",
                      "ආතල්", "ඩාන්ස්", "ට්‍රිප්", "මැච්", "සෙට්"},
    "PleasantExperience": {"කෑම", "කොත්තු", "අයිස්", "සින්දු", "සින්දුව", "ෆිල්ම්",
                           "ෆොටෝ", "වීඩියෝ", "බැලුවා", "දැක්කා", "සපත්තු", "ෂර්ට්",
                           "කාඩ්", "ලැප්", "මල්", "පැටියෙක්", "බස්", "අහස",
                           "නිවාඩු", "නිදහස්", "අවුරුදු", "අවුරුද්දට", "ගමේ", "ඔෆිස්"},

    # Sad frames
    "SadEmotion": {"දුක", "දුකක්", "දුක්", "දුකෙන්", "පාලුයි", "පාළුයි",
                   "අසරණයි", "කළකිරීමක්", "දුකයි"},
    "LossExperience": {"පාලු", "පාලුව", "තනි", "හුදකලා", "ජීවත්",
                       "ඉපදුණේ", "අතරමං", "තැනක්", "හිස්තැනක්"},
    "SufferingExperience": {"වේදනාව", "රිදුම", "කඳුළු", "පපුව", "බරක්",
                            "හුස්ම", "උසුලන්න"},
    "DespairState": {"අවාසනාවන්ත", "අසරණ", "වැටෙන", "වැටෙනවා", "පැරදුණා",
                     "බැරි", "සීමාවක්"},
    "EmotionalLonging": {"හිතේ", "හදවතේ", "හදවත", "මතකයක්", "ආදරේ", "හීන",
                         "හීනයක්ම", "අතීතය", "ලෝකේ", "ලෝකෙටම", "ලෝකයම", "සතුටක්ම"},

    # Angry frames
    "PhysicalHarm": {"ගහනවා", "මරනවා", "කඩනවා", "කපනවා", "ගහලා", "මරල",
                     "ගහගන්න", "ගහන්නෙ", "ගැහුවොත්", "උගුල්ලනවා",
                     "අල්ලගන්නවා", "ගලවනවා", "බස්සනවා"},
    "VerbalInsult": {"පරයා", "බල්ලෙක්", "ගොනා", "කාලකණ්ණියෙක්", "වසල",
                     "කැත", "සවුත්තු", "ජරා", "සෝබන", "කුණු", "අහංකාරකම"},
    "ThreatAction": {"පලයන්", "වහපන්", "නවත්තපන්", "බලපන්", "පෙන්නන්නම්",
                     "පන්නන්න", "දැනගන්", "අතාරින්නෙ", "වරෙන්", "වරෙන්කෝ"},
    "Destruction": {"කැඩුවා", "කැඩිච්ච", "විනාශ"},
    "HostileExclamation": {"යකෝ", "යකා", "ඒයි"},
}

# Map emotion -> General context frame name
GENERAL_FRAMES = {
    "Happy": "GeneralHappyContext",
    "Sad": "GeneralSadContext",
    "Angry": "GeneralAngryContext",
}

# Map frame -> parent emotion
FRAME_EMOTION_MAP = {}
for frame_name, seeds in FRAME_SEEDS.items():
    if frame_name in ["PositiveEmotion", "PositiveQuality", "Achievement",
                       "SocialBonding", "PleasantExperience", "GeneralHappyContext"]:
        FRAME_EMOTION_MAP[frame_name] = "Happy"
    elif frame_name in ["SadEmotion", "LossExperience", "SufferingExperience",
                         "DespairState", "EmotionalLonging", "GeneralSadContext"]:
        FRAME_EMOTION_MAP[frame_name] = "Sad"
    elif frame_name in ["PhysicalHarm", "VerbalInsult", "ThreatAction",
                         "Destruction", "HostileExclamation", "GeneralAngryContext"]:
        FRAME_EMOTION_MAP[frame_name] = "Angry"


def assign_frame(word, emotion):
    """Assign a word to the best matching frame based on seed membership."""
    # Check if it's a modifier or role marker (exclude from frames)
    if word in ALL_MODIFIERS or word in KNOWN_ROLE_MARKERS:
        return None

    # Check seed sets
    for frame_name, seeds in FRAME_SEEDS.items():
        if word in seeds and FRAME_EMOTION_MAP.get(frame_name) == emotion:
            return frame_name

    # Default: General context frame for this emotion
    return GENERAL_FRAMES.get(emotion, None)


def extract_lexicon():
    print(f"Loading data from {data_file}...")
    if not os.path.exists(data_file):
        print("Data file not found!")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab_by_emotion = {
        "Happy": Counter(),
        "Sad": Counter(),
        "Angry": Counter(),
    }

    all_words_counter = Counter()
    modifier_stats = {"negation": Counter(), "intensifier": Counter(),
                      "diminisher": Counter(), "connective": Counter()}

    print("Processing samples...")
    for item in data:
        text = item.get("text", "")
        emotion = item.get("expected", "")

        if emotion in vocab_by_emotion:
            tokens = indic_tokenize.trivial_tokenize(text)
            valid_tokens = [t for t in tokens if len(t) >= MIN_LENGTH]

            vocab_by_emotion[emotion].update(valid_tokens)
            all_words_counter.update(valid_tokens)

            # Track modifier usage
            for t in valid_tokens:
                if t in KNOWN_NEGATION:
                    modifier_stats["negation"][t] += 1
                if t in KNOWN_INTENSIFIERS:
                    modifier_stats["intensifier"][t] += 1
                if t in KNOWN_DIMINISHERS:
                    modifier_stats["diminisher"][t] += 1
                if t in KNOWN_CONNECTIVES:
                    modifier_stats["connective"][t] += 1

    # 1. Get candidates
    candidates = {}
    for emotion in vocab_by_emotion:
        candidates[emotion] = set(
            [w for w, c in vocab_by_emotion[emotion].most_common(TOP_N_CANDIDATES)]
        )

    # 2. Filter for Exclusivity
    exclusive_candidates = {e: set(words) for e, words in candidates.items()}
    emotions = list(candidates.keys())
    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            e1 = emotions[i]
            e2 = emotions[j]
            intersection = candidates[e1].intersection(candidates[e2])
            if intersection:
                print(f"Removing {len(intersection)} overlapping words between {e1} and {e2}")
                exclusive_candidates[e1] -= intersection
                exclusive_candidates[e2] -= intersection

    # 3. Build flat lexicon (backward compatible)
    final_lexicon = {}
    for emotion in emotions:
        sorted_exclusive = sorted(
            list(exclusive_candidates[emotion]),
            key=lambda w: vocab_by_emotion[emotion][w],
            reverse=True
        )
        selected = sorted_exclusive[:FINAL_N]
        final_lexicon[emotion] = selected
        print(f"Selected {len(selected)} exclusive words for {emotion}")

    # Save flat lexicon
    print(f"Saving flat lexicon to {lexicon_file}...")
    with open(lexicon_file, 'w', encoding='utf-8') as f:
        json.dump(final_lexicon, f, ensure_ascii=False, indent=4)

    # 4. Build frame-based lexicon
    print("\nBuilding frame-based lexicon...")

    # Load existing frames.json as template (preserve structure)
    frames_template = {}
    if os.path.exists(frames_file):
        with open(frames_file, 'r', encoding='utf-8') as f:
            frames_template = json.load(f)

    # Initialize frame word lists
    frame_words = {frame_name: [] for frame_name in frames_template}
    for general_frame in GENERAL_FRAMES.values():
        if general_frame not in frame_words:
            frame_words[general_frame] = []

    # Assign each exclusive word to a frame
    for emotion, words in final_lexicon.items():
        for word in words:
            frame_name = assign_frame(word, emotion)
            if frame_name and frame_name in frame_words:
                frame_words[frame_name].append(word)

    # Update frames template with new word lists
    for frame_name, words in frame_words.items():
        if frame_name in frames_template:
            frames_template[frame_name]["words"] = words

    # Save updated frames
    print(f"Saving frame-based lexicon to {frames_file}...")
    with open(frames_file, 'w', encoding='utf-8') as f:
        json.dump(frames_template, f, ensure_ascii=False, indent=4)

    # Print frame statistics
    print("\nFrame distribution:")
    for frame_name, words in frame_words.items():
        print(f"  {frame_name}: {len(words)} words")

    # 5. Print modifier statistics
    print("\nModifier usage in dataset:")
    for mod_type, counts in modifier_stats.items():
        total = sum(counts.values())
        print(f"  {mod_type}: {total} occurrences")
        for word, count in counts.most_common(5):
            print(f"    '{word}': {count}")

    print("\nLexicon extraction complete.")


if __name__ == "__main__":
    extract_lexicon()
