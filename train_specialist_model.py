import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/final_symptom_speciality_dataset.csv")

print("Dataset loaded:", df.shape)
print(df.head())


# =========================
# REMOVE SMALL CLASSES
# =========================

counts = df["Speciality"].value_counts()

valid_classes = counts[counts >= 30].index

df = df[df["Speciality"].isin(valid_classes)]

print("\nFiltered dataset size:", df.shape)
print(df["Speciality"].value_counts())


# =========================
# BALANCE DATASET
# =========================

df_general = df[df["Speciality"] == "General Medicine"].sample(500, random_state=42)

df_other = df[df["Speciality"] != "General Medicine"]

df_balanced = pd.concat([df_general, df_other])

print("\nBalanced dataset size:", df_balanced.shape)
print(df_balanced["Speciality"].value_counts())


# =========================
# TEXT + LABELS
# =========================

X_text = df_balanced["text"].values
y = df_balanced["Speciality"].values


# =========================
# LABEL ENCODER
# =========================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# =========================
# LOAD SENTENCE TRANSFORMER
# =========================

print("\nLoading embedding model...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# GENERATE EMBEDDINGS
# =========================

print("Generating embeddings...")

X_embeddings = embedder.encode(
    X_text,
    batch_size=32,
    show_progress_bar=True
)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)


# =========================
# TRAIN CLASSIFIER
# =========================

print("\nTraining classifier...")

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

clf.fit(X_train, y_train)


# =========================
# EVALUATION
# =========================

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# =========================
# SAVE MODELS
# =========================

joblib.dump(clf, "model/specialist_classifier.pkl")

joblib.dump(label_encoder, "model/label_encoder.pkl")

print("\nModels saved successfully in model/")