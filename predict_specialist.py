import joblib
import os
from sentence_transformers import SentenceTransformer


# -----------------------------
# Get model paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFIER_PATH = os.path.join(BASE_DIR, "model", "specialist_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")


# -----------------------------
# Load models
# -----------------------------

try:
    clf = joblib.load(CLASSIFIER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("✅ ML specialist model loaded")

except Exception as e:

    print("⚠️ ML model not found")
    print("Error:", e)

    clf = None
    embedder = None
    label_encoder = None


# -----------------------------
# Prediction function
# -----------------------------

def predict_specialist(symptoms_text):

    if clf is None or embedder is None:
        return "General Medicine"

    symptoms_text = symptoms_text.lower()

    # Convert text → embedding
    emb = embedder.encode([symptoms_text])

    # Predict class index
    prediction_index = clf.predict(emb)[0]

    # Convert index → specialist name
    specialist = label_encoder.inverse_transform([prediction_index])[0]

    return specialist