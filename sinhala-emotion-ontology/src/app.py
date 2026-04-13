from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.classify import EmotionClassifier
from src.tts_manager import TTSManager
import uvicorn
import os

# Initialize App
app = FastAPI(
    title="Sinhala Emotion Ontology API",
    description="Classifies Sinhala text into Happy, Sad, Angry, or Neutral using Ontology & ML.",
    version="1.0.0"
)

# Mount Static Files Directory securely
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize Classifier and TTS Manager (Global to load once)
classifier = EmotionClassifier()
tts_manager = TTSManager()

class TTSRequest(BaseModel):
    text: str
    model_id: str | None = "v3"

class TTSResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    method: str
    audio_base64: str

class ClassificationResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    method: str
    matched_words: dict | None = None

@app.get("/")
def read_root():
    # Return the Custom Dark Mode Vue/JS interface
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Sinhala Emotion Ontology API. UI not found!"}

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

import base64

@app.post(
    "/tts",
    summary="Classify emotion and generate speech",
    response_model=TTSResponse
)
def synthesize_speech(request: TTSRequest):
    """
    Classifies the emotion of the text and then uses the TTS model to generate speech.
    Returns the JSON response including the categorized emotion alongside a base64 encoded audio string!
    """
    text = request.text
    
    # 1. Predict emotion
    result = classifier.predict(text)
    
    # 2. Generate audio
    model_id = request.model_id if request.model_id else "v3"
    audio_bytes = tts_manager.generate_audio(text, model_id=model_id)
    
    # 3. Encode to base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    return {
        "text": text,
        "emotion": result["label"],
        "confidence": result["confidence"],
        "method": result["method"],
        "audio_base64": audio_b64
    }

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
