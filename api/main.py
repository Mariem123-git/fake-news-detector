# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import time
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Fake News Detection API",
    description="API pour détecter les fausses nouvelles",
    version="1.0.0"
)

# Variables globales pour le modèle et le vectorizer
model = None
vectorizer = None

# Obtenir le chemin absolu du dossier du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier api/
PROJECT_DIR = os.path.dirname(BASE_DIR)  # dossier racine


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

    # Utilisation des chemins absolus
    model_path = os.path.join(PROJECT_DIR, "models", "model.pkl")
    vectorizer_path = os.path.join(PROJECT_DIR, "models", "vectorizer.pkl")

    logger.info(f"Recherche du modèle dans : {model_path}")
    logger.info(f"Recherche du vectorizer dans : {vectorizer_path}")
    logger.info(f"Répertoire du projet : {PROJECT_DIR}")

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            logger.info("✅ Modèle et vectorizer chargés avec succès")

            if hasattr(model, 'classes_'):
                logger.info(f"Classes du modèle: {model.classes_}")
            if hasattr(model, 'predict_proba'):
                logger.info("✅ Modèle supporte predict_proba")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
    else:
        logger.warning(f"⚠️ Modèle non trouvé: {model_path}")
        logger.warning(f"⚠️ Vectorizer non trouvé: {vectorizer_path}")


@app.get("/")
def root():
    return {
        "message": "Fake News Detection API",
        "docs": "/docs",
        "health": "/health"
    }


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
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Veuillez réessayer plus tard."
        )

    if not article.text or len(article.text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Le texte doit contenir au moins 10 caractères."
        )

    try:
        X = vectorizer.transform([article.text])

        proba = model.predict_proba(X)[0]
        real_prob = float(proba[0])
        fake_prob = float(proba[1])

        latency = (time.time() - start_time) * 1000

        logger.info(f"Prédiction - Fake: {fake_prob:.2%} | Latence: {latency:.2f}ms")

        return PredictionResponse(
            prediction="fake" if fake_prob > 0.5 else "real",
            confidence=max(fake_prob, real_prob),
            fake_probability=fake_prob,
            real_probability=real_prob,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/debug")
def debug():
    model_path = os.path.join(PROJECT_DIR, "models", "model.pkl")
    return {
        "project_dir": PROJECT_DIR,
        "model_path": model_path,
        "model_exists": os.path.exists(model_path),
        "files_in_models": os.listdir(os.path.join(PROJECT_DIR, "models"))
                           if os.path.exists(os.path.join(PROJECT_DIR, "models"))
                           else "dossier models inexistant ❌"
    } 
