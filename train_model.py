import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

import evaluate
import numpy as np

# Set model details
MODEL_ID = "distilbert-base-uncased"
OUTPUT_DIR = "./local_model"
TRAIN_DATA_PATH = "data/medical_tc_train.csv"
TEST_DATA_PATH = "data/medical_tc_test.csv"

# Label mappings (consistent with rule-based system)
LABEL_MAP = {
    0: "Routine/Stable",
    1: "Urgent/Critical"
}

def load_data():
    """Loads the datasets for training and testing with optimized hardcoded columns."""
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Datasets not found at {TRAIN_DATA_PATH} or {TEST_DATA_PATH}.")
        return None, None
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # 1. Fast rename using mapping dictionary
    rename_map = {'medical_abstract': 'text', 'condition_label': 'label'}
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    if 'text' not in train_df.columns or 'label' not in train_df.columns:
        print("Error: Missing expected columns 'medical_abstract' or 'condition_label'.")
        return None, None

    # 2. Fast vectorized label mapping: 1-2 -> 0 (Routine/Stable), 3-5 -> 1 (Urgent/Critical)
    train_df['label'] = (train_df['label'] >= 3).astype(int)
    test_df['label'] = (test_df['label'] >= 3).astype(int)

    # Drop rows with missing values
    train_df = train_df.dropna(subset=['text', 'label'])
    test_df = test_df.dropna(subset=['text', 'label'])
    
    return train_df, test_df

class ClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train():
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        print("Exiting training script.")
        return

    print("🚀 Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label={0: "Routine/Stable", 1: "Urgent/Critical"},
        label2id={"Routine/Stable": 0, "Urgent/Critical": 1}
    )

    print("📊 Tokenizing datasets...")
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = ClinicalDataset(train_encodings, train_df['label'].tolist())
    test_dataset = ClinicalDataset(test_encodings, test_df['label'].tolist())

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_strategy="no",
        eval_strategy="no",
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("🔥 Starting training...")
    trainer.train()

    print("🧪 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Eval Results: {eval_results}")

    print(f"💾 Saving fine-tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Training complete. Model is ready for integration.")

if __name__ == "__main__":
    train()
