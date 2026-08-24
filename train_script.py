# import re
# import numpy as np
# import pandas as pd
# from bs4 import BeautifulSoup
# from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, ClassLabel
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
# from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer


# DATASET_PATH = "stanfordnlp/imdb"
# MAX_LENGTH = 512
# MODEL_CHECKPOINT = "distilbert-base-uncased"

# # ======================Functions Start========================================================================================

# # =================clean_text=======================
# def clean_text(text):
#     text = BeautifulSoup(text, "html.parser").get_text(" ")
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# # ================tokenize_function==================
# def tokenize_function(examples):
#     return tokenizer(
#         examples["text"],
#         truncation=True,
#         max_length=MAX_LENGTH
#     )


# # ================compute_metrics==================
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred

#     predictions = np.argmax(logits, axis=-1)

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         labels,
#         predictions,
#         average="binary"
#     )

#     accuracy = accuracy_score(
#         labels,
#         predictions
#     )

#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }

# # ======================functions End========================================================================================


# dataset = load_dataset(DATASET_PATH)
# full_dataset = concatenate_datasets([
#     dataset["train"],
#     dataset["test"]
# ])

# df = full_dataset.to_pandas()
# df.drop_duplicates(keep = 'first',inplace=True)
# df['text'] = df['text'].map(clean_text)

# num_rows = df.shape[0]
# test_size = int(num_rows *0.2)

# full_dataset = Dataset.from_pandas(df[['text','label']],preserve_index=False)
# full_dataset = full_dataset.cast_column(
#     "label",
#     ClassLabel(names=["neg", "pos"])
# )

# train_test_split = full_dataset.train_test_split(
#     test_size=test_size,
#     stratify_by_column="label",
#     seed=42
# )


# train_dataset = train_test_split["train"]
# temp_dataset = train_test_split["test"]


# validation_test_split = temp_dataset.train_test_split(
#     test_size=test_size//2,
#     stratify_by_column="label",
#     seed=42
# )

# validation_dataset = validation_test_split["train"]
# test_dataset = validation_test_split["test"]

# dataset = DatasetDict({
#     "train": train_dataset,
#     "validation": validation_dataset,
#     "test": test_dataset
# })


# tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
# tokenized_dataset = dataset.map(
#     tokenize_function,
#     batched=True
# )


# data_collator = DataCollatorWithPadding(
#     tokenizer=tokenizer
# )


# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_CHECKPOINT,
#     num_labels=2,
#     id2label={
#         0: "NEGATIVE",
#         1: "POSITIVE"
#     },
#     label2id={
#         "NEGATIVE": 0,
#         "POSITIVE": 1
#     }
# )

# training_args = TrainingArguments(
#     output_dir="sentiment_model",

#     # Maximum training epochs
#     num_train_epochs=2,

#     learning_rate=2e-5,

#     # Batch size
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=32,

#     # No gradient accumulation
#     gradient_accumulation_steps=1,

#     # Optimizer
#     optim="adamw_torch",

#     # Learning rate scheduler
#     lr_scheduler_type="cosine",
#     # 5%
#     warmup_steps=1240,

#     # Mixed precision
#     fp16=True,
    
#     # Evaluation
#     eval_strategy="epoch",


#     # Saving
#     save_strategy="epoch",
#     save_total_limit=2,
#     group_by_length=True,

#     # Load best model
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,

#     # Logging
#     logging_steps=100,

#     # Reproducibility
#     seed=42
# )


# early_stopping = EarlyStoppingCallback(
#     early_stopping_patience=2
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     # callbacks=[early_stopping]
# )

# train_result = trainer.train()

# print(train_result.metrics)

# print("Best checkpoint:", trainer.state.best_model_checkpoint)
# print("Best metric:", trainer.state.best_metric)

# test_results = trainer.evaluate(
#     eval_dataset=tokenized_dataset["test"]
# )

# print(test_results)


# predictions_output = trainer.predict(
#     tokenized_dataset["test"]
# )

# logits = predictions_output.predictions

# predictions = np.argmax(
#     logits,
#     axis=-1
# )

# true_labels = predictions_output.label_ids

# print(
#     classification_report(
#         true_labels,
#         predictions,
#         target_names=["NEGATIVE", "POSITIVE"]
#     )
# )

# cm = confusion_matrix(
#     true_labels,
#     predictions
# )

# print(cm)



import argparse
import os
import re

import numpy as np
import torch
from bs4 import BeautifulSoup
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DATASET_PATH = "stanfordnlp/imdb"
MAX_LENGTH = 512
MODEL_CHECKPOINT = "distilbert-base-uncased"

OUTPUT_DIR = "sentiment_model"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final")

# Speed flags — safe no-ops on GPUs that don't support them.
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ======================Functions Start=========================================


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def build_dataset():
    """Load, clean, and split the IMDB dataset. Same logic as before, unchanged."""
    raw = load_dataset(DATASET_PATH)
    full_dataset = concatenate_datasets([raw["train"], raw["test"]])

    df = full_dataset.to_pandas()
    df.drop_duplicates(keep="first", inplace=True)
    df["text"] = df["text"].map(clean_text)

    num_rows = df.shape[0]
    test_size = int(num_rows * 0.2)

    full_dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    full_dataset = full_dataset.cast_column("label", ClassLabel(names=["neg", "pos"]))

    train_test_split = full_dataset.train_test_split(
        test_size=test_size, stratify_by_column="label", seed=42
    )
    train_dataset = train_test_split["train"]
    temp_dataset = train_test_split["test"]

    validation_test_split = temp_dataset.train_test_split(
        test_size=test_size // 2, stratify_by_column="label", seed=42
    )
    validation_dataset = validation_test_split["train"]
    test_dataset = validation_test_split["test"]

    return DatasetDict(
        {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}
    )


def get_tokenized_dataset(tokenizer):
    dataset = build_dataset()
    return dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)


def build_training_args(resume_epochs=None):
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=resume_epochs if resume_epochs else 2,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=1240,
        # Prefer bf16 on Ampere+ GPUs (RTX 30/40-series); fall back to fp16 otherwise.
        bf16=USE_BF16,
        fp16=not USE_BF16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=42,
    )


def load_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        attn_implementation="sdpa",
    )


def run_test_evaluation(trainer, tokenized_dataset):
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(test_results)

    predictions_output = trainer.predict(tokenized_dataset["test"])
    logits = predictions_output.predictions
    predictions = np.argmax(logits, axis=-1)
    true_labels = predictions_output.label_ids

    print(
        classification_report(
            true_labels, predictions, target_names=["NEGATIVE", "POSITIVE"]
        )
    )
    print(confusion_matrix(true_labels, predictions))


# ======================Functions End=========================================


def train(resume_checkpoint=None, total_epochs=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized_dataset = get_tokenized_dataset(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = load_model()
    training_args = build_training_args(resume_epochs=total_epochs)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[early_stopping]
    )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(train_result.metrics)
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric:", trainer.state.best_metric)

    # Save the final selected model + tokenizer so evaluate/predict never need to retrain.
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Final model saved to {FINAL_MODEL_DIR}")

    run_test_evaluation(trainer, tokenized_dataset)


def evaluate():
    if not os.path.isdir(FINAL_MODEL_DIR):
        raise FileNotFoundError(
            f"No saved model found at {FINAL_MODEL_DIR}. Run `--mode train` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)
    tokenized_dataset = get_tokenized_dataset(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)

    # eval-only args: no training happens, this just configures batch size / precision
    eval_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=32,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    run_test_evaluation(trainer, tokenized_dataset)


def predict(text):
    if not os.path.isdir(FINAL_MODEL_DIR):
        raise FileNotFoundError(
            f"No saved model found at {FINAL_MODEL_DIR}. Run `--mode train` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
    model.eval()

    inputs = tokenizer(
        clean_text(text), truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(np.argmax(logits.numpy(), axis=-1)[0])
    label = model.config.id2label[pred_id]
    print(f"Prediction: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        default="train",
        help="train: fine-tune and save model. evaluate: load saved model, run on test set. predict: run one example.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint dir, e.g. sentiment_model/checkpoint-4958",
    )
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=None,
        help="Total target epoch count when resuming (not additional epochs).",
    )
    parser.add_argument("--text", default=None, help="Text to classify in predict mode")

    parser.add_argument(
        "--text_file",
        default=None,
        help="Path to a .txt file containing the review to classify (avoids shell quoting "
        "issues with quotes/apostrophes in --text).",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(resume_checkpoint=args.resume_from_checkpoint, total_epochs=args.total_epochs)
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "predict":
        if args.text_file:
            with open(args.text_file, "r", encoding="utf-8") as f:
                input_text = f.read()
        elif args.text:
            input_text = args.text
        else:
            raise ValueError("Provide --text or --text_file for predict mode")
        predict(input_text)