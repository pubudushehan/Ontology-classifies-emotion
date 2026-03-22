import sys
from src.classify import EmotionClassifier

classifier = EmotionClassifier()
test_sentences = [
    "ජීවිතේ පොඩි පොඩි දේවල් වලින් සතුටු වෙන්න පුළුවන් නම් ඒ ඇති.",
    "මට හරිම දුකයි",
    "මම කේන්තියෙන් ඉන්නේ"
]

for t in test_sentences:
    print(f"Text: {t} -> {classifier.predict(t)}")
