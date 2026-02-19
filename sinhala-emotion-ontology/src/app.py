from fastapi import FastAPI, Query
from pydantic import BaseModel
from src.classify import EmotionClassifier
import uvicorn
import os

# Initialize App
app = FastAPI(
    title="Sinhala Emotion Ontology API",
    description="Classifies Sinhala text into Happy, Sad, Angry, or Neutral using Ontology & ML.",
    version="1.0.0"
)

# Initialize Classifier (Global to load once)
classifier = EmotionClassifier()

class ClassificationResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    method: str
    matched_words: dict | None = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Sinhala Emotion Ontology API. Visit /docs for Swagger UI."}

@app.get("/classify", response_model=ClassificationResponse)
def classify_text(text: str = Query(..., description="Sinhala sentence to classify")):
    """
    Classifies the input text.
    """
    result = classifier.predict(text)
    return {
        "text": text,
        "emotion": result["label"],
        "confidence": result["confidence"],
        "method": result["method"], # "Ontology" or "ML (LaBSE)"
        "matched_words": result.get("matched_words")
    }

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
