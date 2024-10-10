print("Init...")
import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset
from torch.optim import AdamW  # PyTorch AdamW implementation
from torch.utils.data import random_split
from time import sleep
import pytest

print("Loading...")

# Paths for datasets and models
DATASET_DIR = "./datasets"
MODEL_DIR = "./saved_models"
MODEL_NAME = "bert-base-uncased"

# Function to load and preprocess each dataset
def load_datasets(dataset_dir):
    print("Loading Datasets...")
    datasets = {}
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(dataset_dir, filename), 'r', encoding='utf-8') as file:
                raw_text = file.read()

            # Split by the <S> and <E> tags
            samples = [text.strip() for text in raw_text.split('<E>') if '<S>' in text]
            samples = [text.replace('<S>', '').strip() for text in samples]

            datasets[filename] = samples
    return datasets

# Function to tokenize datasets and prepare labels for MLM
def tokenize_and_mask(examples, tokenizer):
    print("Tokenize and Masking...")
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    # Create labels by copying input_ids
    labels = tokenized_inputs["input_ids"].copy()

    # Replace 15% of the tokens with [MASK] token
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            prob = torch.rand(1).item()
            # Mask with a probability of 15%
            if prob < 0.15:
                tokenized_inputs["input_ids"][i][j] = tokenizer.mask_token_id

    tokenized_inputs["labels"] = labels  # Assign the labels
    return tokenized_inputs

# Fine-tune BERT models and save them
def fine_tune_models(datasets, model_dir):
    print("Fine Tuning Model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    for dataset_name, samples in datasets.items():
        print(f"Training model on dataset: {dataset_name}")
        
        # Convert samples into Hugging Face dataset
        hf_dataset = Dataset.from_dict({"text": samples})

        # Tokenize and create masked inputs and labels
        tokenized_dataset = hf_dataset.map(lambda x: tokenize_and_mask(x, tokenizer), batched=True)

        # Split into train/validation (90% train, 10% val)
        train_size = int(0.9 * len(tokenized_dataset))
        train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, len(tokenized_dataset) - train_size])

        # Load BERT model for Masked Language Modeling (MLM)
        model = BertForMaskedLM.from_pretrained(MODEL_NAME)

        # Define training arguments, using torch.optim.AdamW instead of transformers.AdamW
        training_args = TrainingArguments(
            output_dir=f"{model_dir}/{dataset_name}_model",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_steps=10_000,
            save_total_limit=2,
            optim="adamw_torch"  # Use PyTorch AdamW
        )

        # Trainer API
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Fine-tune the model
        trainer.train()

        # Save the model and tokenizer
        model.save_pretrained(f"{model_dir}/{dataset_name}_model")
        tokenizer.save_pretrained(f"{model_dir}/{dataset_name}_model")

# Function to load a saved model and use it for inference
def load_and_use_model(model_dir, model_name, input_text):
    # Ensure that the full path to the local model directory is provided
    model_path = os.path.join(model_dir, model_name)

    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)

        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt")

        # Predict masked words
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Decode predictions
        predicted_text = tokenizer.decode(predictions[0])
        return predicted_text
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Ready!")
    try:
        # Step 1: Load and preprocess datasets
        #datasets = load_datasets(DATASET_DIR)

        # Step 2: Fine-tune models on each dataset
        #fine_tune_models(datasets, MODEL_DIR)

        # Step 3: Example of using the saved models later
        test_input = "You should sell your stocks after [MASK]"
        model_name = "Center_Political_sample.txt_model"  # Replace with the actual dataset filename used
        predicted_text = load_and_use_model(MODEL_DIR, model_name, test_input)
    
        print(f"Predicted text: {predicted_text}")
    except Exception as e:
        print("ERROR: " + e)
        sleep(100)
        input("Press Enter to continue...")
