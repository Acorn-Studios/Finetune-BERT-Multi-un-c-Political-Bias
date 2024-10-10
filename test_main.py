import os
import pytest
from unittest.mock import patch, MagicMock
import torch
from main import *

# Test loading datasets
@patch("os.listdir")
@patch("builtins.open", create=True)
def test_load_datasets(mock_open, mock_listdir):
    mock_listdir.return_value = ["dataset1.txt"]
    mock_open.return_value.__enter__.return_value.read.return_value = "<S> Sample text <E>"
    
    datasets = load_datasets("./datasets")
    
    assert len(datasets) == 1
    assert "dataset1.txt" in datasets
    assert datasets["dataset1.txt"] == ["Sample text"]

# Test tokenization and masking
def test_tokenize_and_mask():
    tokenizer = MagicMock()
    tokenizer.mask_token_id = 103  # BERT [MASK] token ID
    examples = {"text": ["You should sell your stocks after the market [MASK]"]}
    
    tokenized = {
        "input_ids": [[101, 2002, 2064, 3749, 2015, 2115, 15710, 2044, 1996, 3006, 103]],
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    }
    tokenizer.return_value = tokenized
    
    result = tokenize_and_mask(examples, tokenizer)
    
    assert "input_ids" in result
    assert "labels" in result
    assert len(result["input_ids"]) == len(result["labels"])

# Test fine-tuning models (mock the actual training process)
@patch("transformers.Trainer.train")
@patch("transformers.BertForMaskedLM.from_pretrained")
@patch("transformers.BertTokenizer.from_pretrained")
def test_fine_tune_models(mock_tokenizer, mock_model, mock_train):
    datasets = {"dataset1.txt": ["Sample text"]}
    tokenizer_mock = MagicMock()
    model_mock = MagicMock()
    
    mock_tokenizer.return_value = tokenizer_mock
    mock_model.return_value = model_mock
    mock_train.return_value = None  # Mocking out the actual training process
    
    fine_tune_models(datasets, "./saved_models")
    
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
    mock_train.assert_called_once()

# Test loading and using the model (mock model inference)
@patch("transformers.BertForMaskedLM.from_pretrained")
@patch("transformers.BertTokenizer.from_pretrained")
def test_load_and_use_model(mock_tokenizer, mock_model):
    tokenizer_mock = MagicMock()
    model_mock = MagicMock()
    
    mock_tokenizer.return_value = tokenizer_mock
    mock_model.return_value = model_mock
    
    # Mock tokenizer input and model output
    tokenizer_mock.return_tensors = "pt"
    tokenizer_mock.decode.return_value = "You should sell your stocks after the market crashes"
    
    model_mock.__call__.return_value.logits = torch.tensor([[[100, 200, 300]]])  # Mocked logits

    result = load_and_use_model("./saved_models", "dataset1.txt_model", "You should sell your stocks after [MASK]")
    
    assert result == "You should sell your stocks after the market crashes"
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
