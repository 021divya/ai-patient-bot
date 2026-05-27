import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch

#  Load dataset
df = pd.read_csv("data/final_symptom_speciality_dataset.csv")

# Encode labels
label_map = {label: idx for idx, label in enumerate(df["Speciality"].unique())}
df["label"] = df["Speciality"].map(label_map)

# Save label map (IMPORTANT for inference)
with open("model/label_map.json", "w") as f:
    json.dump(label_map, f)

#  Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_enc = tokenizer(train_texts, truncation=True, padding=True)
val_enc = tokenizer(val_texts, truncation=True, padding=True)

#  Dataset class
class SymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SymptomDataset(train_enc, train_labels)
val_dataset = SymptomDataset(val_enc, val_labels)

#  Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_map)
)

#  Training arguments
training_args = TrainingArguments(
    output_dir="model",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs",
    save_total_limit=1,
    report_to="none"
)

#  Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

#  Train
trainer.train()

# Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("✅ Model training completed and saved.")
