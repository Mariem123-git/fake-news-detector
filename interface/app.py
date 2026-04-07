# interface/app.py
import streamlit as st
import requests
import os
import time
import random

# Configuration de la page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="",
    layout="wide"
)

# Titre principal
st.title(" Fake News Detection System")
st.markdown("Détectez les fausses nouvelles avec l'intelligence artificielle")

# URL de l'API (locale pour l'instant)
API_URL = "https://fake-news-detector-mq45.onrender.com/"

# Sidebar - Informations
with st.sidebar:
    st.header("ℹ️ État du système")

    # Vérification de la connexion à l'API
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                st.success(" API connectée - Modèle chargé")
            else:
                st.warning(" API connectée - Mode simulation (en attente du modèle final)")
        else:
            st.error(" API non disponible")
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API")
        st.info("💡 Lancez l'API avec : uvicorn api.main:app --reload --port 8000")
    except Exception as e:
        st.error(f"❌ Erreur: {e}")

    st.markdown("---")
    st.markdown("### Comment ça fonctionne ?")
    st.markdown("""
    1. Entrez le texte d'un article
    2. Cliquez sur Analyser
    3. L'IA vous dit si c'est une fake news
    """)

    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    - **Modèle**: Logistic Regression (Tuned)
    - **Dataset**: 44k+ articles
    - **API**: FastAPI
    - **Interface**: Streamlit
    """)

# Zone principale
st.subheader(" Entrez l'article à analyser")

article = st.text_area(
    "",
    height=250,
    placeholder="Collez le texte de l'article ici... (minimum 10 caractères)"
)

# Bouton d'analyse
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyser = st.button(" Analyser", type="primary", use_container_width=True)

# Résultats
if analyser:
    if len(article.strip()) < 10:
        st.warning("⚠️ Veuillez entrer au moins 10 caractères")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                start_time = time.time()

                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": article},
                    timeout=30
                )

                elapsed_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Analyse terminée !")

                    # Affichage des résultats
                    col1, col2, col3, col4 = st.columns(4)

                    # Prédiction
                    if result["prediction"] == "fake":
                        col1.metric("📰 Prédiction", "🔴 FAKE NEWS", delta="Attention")
                    else:
                        col1.metric("📰 Prédiction", "✅ VRAI", delta="Confiance")

                    # Confiance
                    col2.metric("🎯 Confiance", f"{result['confidence']:.1%}")

                    # Probabilités
                    col3.metric("📊 Probabilité FAKE", f"{result['fake_probability']:.1%}")
                    col4.metric("📊 Probabilité REAL", f"{result['real_probability']:.1%}")

                    # Barre de progression
                    st.subheader("Détail de la prédiction")

                    if result["prediction"] == "fake":
                        st.progress(
                            result["fake_probability"],
                            text=f"🔴 Probabilité que ce soit une FAKE NEWS : {result['fake_probability']:.1%}"
                        )
                    else:
                        st.progress(
                            result["real_probability"],
                            text=f"✅ Probabilité que ce soit VRAI : {result['real_probability']:.1%}"
                        )

                    # Métadonnées
                    st.caption(
                        f"⏱️ Temps de réponse API : {result['latency_ms']:.0f} ms | Temps total : {elapsed_time:.0f} ms")

                elif response.status_code == 503:
                    st.error("❌ Modèle non disponible. Veuillez réessayer plus tard.")
                    st.info("ℹ️ L'API est en attente du modèle final. Contactez P2.")
                else:
                    st.error(f"❌ Erreur {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                st.error("❌ Délai d'attente dépassé. Veuillez réessayer.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API. Vérifiez que l'API est lancée.")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")

# Footer
st.markdown("---")
st.markdown(" **Projet Machine Learning - Fake News Detection**")

