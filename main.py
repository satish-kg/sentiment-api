import pickle
from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd # Although not strictly needed, good practice if using DataFrame logic

# --- 1. Load Assets ---
# Load the trained vectorizer and model once when the app starts. "only for save"
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        VECTORIZER = pickle.load(f)
    with open('sentiment_model.pkl', 'rb') as f:
        MODEL = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model or Vectorizer files not found. Ensure they are in the project root.")

# --- 2. Initialize FastAPI ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts sentiment (-1, 0, 1) for a given text using LinearSVC.",
    version="1.0.0"
)

origins = [
    "https://senfiment-analyzer.netlify.app", 
]

app.add_middleware(
    CORSMiddleware,
    #allow_origins=origins, 
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Define the Input Schema (Pydantic Model) ---
# This ensures the API only accepts a JSON body with a 'text' string
class TextToPredict(BaseModel):
    text: str
    
# Map the numerical outputs back to human-readable labels
LABEL_MAP = {-1.0: "Negative", 0.0: "Neutral", 1.0: "Positive"}

# --- 4. Define the Prediction Endpoint ---
@app.post("/predict_sentiment")
def predict_sentiment(data: TextToPredict):
    """
    Accepts text and returns the predicted sentiment.
    """
    # 1. Get the text from the validated Pydantic model
    input_text = data.text

    # 2. Preprocess/Vectorize the text using the loaded vectorizer
    # NOTE: The input must be a list/series of strings for transform()
    text_vectorized = VECTORIZER.transform([input_text])

    # 3. Make the prediction
    prediction = MODEL.predict(text_vectorized)
    
    # 4. Map the numerical prediction to the text label
    predicted_label = prediction[0]
    sentiment_text = LABEL_MAP.get(predicted_label, "Unknown")
    
    # 5. Return the result as JSON
    return {
        "input_text": input_text,
        "prediction_code": predicted_label,
        "sentiment": sentiment_text
    }

# --- 5. Simple Health Check Endpoint (Optional but Recommended) ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sentiment API is running"}
