# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import time
import logging
import re
import string
import nltk

# Télécharger les ressources NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prétraitement — identique à Colab !
stop = set(stopwords.words('english'))
stop.update(string.punctuation)
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'reuters', '', text)
    text = re.sub(r'\d+', '', text)
    text = " ".join(
        word.strip() for word in text.split()
        if word.strip() not in stop
    )
    tokens = word_tokenize(text)
    text = " ".join(lemmatizer.lemmatize(w) for w in tokens)
    return text

# Création de l'application FastAPI
app = FastAPI(
    title="Fake News Detection API",
    description="API pour détecter les fausses nouvelles",
    version="1.0.0"
)

model = None
vectorizer = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

class Article(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float
    latency_ms: float

@app.on_event("startup")
def load_model():
    global model, vectorizer
    model_path = os.path.join(PROJECT_DIR, "models", "model(1).pkl")
    vectorizer_path = os.path.join(PROJECT_DIR, "models", "vectorizer(1).pkl")

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            logger.info("✅ Modèle et vectorizer chargés avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
    else:
        logger.warning(f"⚠️ Modèle non trouvé")

@app.get("/")
def root():
    return {"message": "Fake News Detection API", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(article: Article):
    start_time = time.time()

    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    if not article.text or len(article.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Texte trop court.")

    try:
        # ✅ Prétraitement avant prédiction
        clean = preprocess(article.text)
        logger.info(f"Texte nettoyé : {clean[:100]}...")

        X = vectorizer.transform([clean])

        proba = model.predict_proba(X)[0]
        real_prob = float(proba[0])
        fake_prob = float(proba[1])

        latency = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction="fake" if fake_prob > 0.5 else "real",
            confidence=max(fake_prob, real_prob),
            fake_probability=fake_prob,
            real_probability=real_prob,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")
