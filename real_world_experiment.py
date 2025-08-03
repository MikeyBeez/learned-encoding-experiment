#!/usr/bin/env python3
"""
Real-World Validation of Learned Encoding

This script implements the core models in PyTorch for real-world
end-to-end training and validation on standard datasets.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
import os
import requests
import zipfile
import io
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import Counter, OrderedDict
import time
import math
from transformers import AutoTokenizer

@dataclass
class ModelConfig:
    """Configuration for the real-world models."""
    vocab_size: int = 10000
    # For traditional model
    traditional_embedding_dim: int = 128
    # For both models (compressed dimension)
    compressed_dim: int = 16  # 8:1 compression
    # Transformer params
    num_layers: int = 2
    hidden_dim: int = 64
    nhead: int = 4


class SimpleAutoencoder(nn.Module):
    """
    Traditional autoencoder for learning token representations.
    This is pre-trained separately.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, compressed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_linear = nn.Linear(embedding_dim, compressed_dim)
        self.relu = nn.ReLU()
        self.decoder_linear = nn.Linear(compressed_dim, embedding_dim)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encodes token IDs to a compressed representation."""
        x = self.embedding(token_ids)
        x = self.encoder_linear(x)
        x = self.relu(x)
        return x

    def decode(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decodes from compressed representation back to embedding space."""
        x = self.decoder_linear(compressed)
        return x

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass for pre-training the autoencoder."""
        original_embeddings = self.embedding(token_ids)
        compressed = self.encode(token_ids)
        reconstructed = self.decode(compressed)
        return reconstructed, original_embeddings


class LearnedEncodingModel(nn.Module):
    """
    Model that learns encodings during generation training.
    This is the key innovation.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Key innovation: A learnable token encoder that maps a large vocabulary
        # directly to a compressed, low-dimensional space.
        self.token_encoder = nn.Embedding(config.vocab_size, config.compressed_dim)

        # Standard Transformer layers that operate on the compressed embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.compressed_dim,
            nhead=config.nhead,
            dim_feedforward=config.hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection to predict the next token in the vocabulary
        self.output_projection = nn.Linear(config.compressed_dim, config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for end-to-end training."""
        # 1. Encode tokens directly into the compressed embedding space.
        x = self.token_encoder(token_ids)

        # 2. Pass the compressed sequence through the transformer.
        x = self.transformer_encoder(x)

        # 3. Project the output back to the vocabulary space for prediction.
        logits = self.output_projection(x)
        return logits


class TraditionalModel(nn.Module):
    """
    Traditional model using a pre-trained autoencoder.
    The autoencoder's weights are frozen during training.
    """
    def __init__(self, config: ModelConfig, autoencoder: SimpleAutoencoder):
        super().__init__()
        self.config = config
        self.autoencoder = autoencoder

        # Standard Transformer layers, same as the learned model for fair comparison.
        # Operates on the compressed dimension from the autoencoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.compressed_dim,
            nhead=config.nhead,
            dim_feedforward=config.hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection, same as the learned model.
        self.output_projection = nn.Linear(config.compressed_dim, config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass using the frozen, pre-trained autoencoder."""
        # 1. Get compressed embeddings from the autoencoder.
        # We use no_grad() to ensure the autoencoder is not trained.
        with torch.no_grad():
            x = self.autoencoder.encode(token_ids)

        # 2. Pass the frozen embeddings through the transformer.
        x = self.transformer_encoder(x)

        # 3. Project to vocabulary space.
        logits = self.output_projection(x)
        return logits


import os
import requests
import zipfile
import io
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

class LanguageModelDataset(Dataset):
    """Custom PyTorch Dataset for language modeling from pre-chunked data."""
    def __init__(self, split_data):
        self.split_data = split_data

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, i):
        item = self.split_data[i]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        # For language modeling, the input and target are shifted by one token.
        inputs = input_ids[:-1].clone()
        targets = input_ids[1:].clone()
        return inputs, targets

def get_dataloaders(dataset_name='wikitext', config_name='wikitext-2-raw-v1', batch_size=32, seq_len=35):
    """Returns train, validation, and test dataloaders using a Hugging Face tokenizer."""
    print(f"Loading {dataset_name} dataset and tokenizer...")

    # 1. Load dataset and a pre-trained tokenizer
    # Using a smaller subset of the data for faster verification runs.
    from datasets import DatasetDict
    train_split = load_dataset(dataset_name, config_name, split='train[:10%]')
    val_split = load_dataset(dataset_name, config_name, split='validation[:10%]')
    test_split = load_dataset(dataset_name, config_name, split='test')
    dataset = DatasetDict({
        'train': train_split,
        'validation': val_split,
        'test': test_split
    })
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Pre-tokenize the entire dataset
    def tokenize_function(examples):
        texts = [doc for doc in examples["text"] if doc is not None and isinstance(doc, str)]
        return tokenizer(texts)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Running tokenizer on dataset"
    )

    # 3. Concatenate tokenized texts and chunk them into blocks of `seq_len`
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // seq_len) * seq_len
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {seq_len}"
    )

    # 4. Create PyTorch Datasets and DataLoaders
    train_dataset = LanguageModelDataset(lm_datasets['train'])
    val_dataset = LanguageModelDataset(lm_datasets['validation'])
    test_dataset = LanguageModelDataset(lm_datasets['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    vocab_size = tokenizer.vocab_size
    print(f"✅ DataLoaders created successfully with vocab size {vocab_size}.")
    return train_loader, val_loader, test_loader, vocab_size

def train_autoencoder(autoencoder, train_loader, epochs=5, lr=0.001):
    """Pre-trains the SimpleAutoencoder."""
    print("🤖 Pre-training Autoencoder...")
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    autoencoder.train()

    for epoch in range(epochs):
        total_loss = 0
        for i, (data, _) in enumerate(train_loader):
            # We only need the input data for autoencoder pre-training
            optimizer.zero_grad()
            reconstructed, original = autoencoder(data)
            loss = criterion(reconstructed, original)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"   Autoencoder Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("✅ Autoencoder pre-training complete!")
    return autoencoder

def train_epoch(model, dataloader, optimizer, criterion, vocab_size):
    """Handles a single training epoch for a language model."""
    model.train()
    total_loss = 0
    for i, (data, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        # Reshape for loss function
        output_flat = output.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = criterion(output_flat, targets_flat)
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, vocab_size):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            output = model(data)
            output_flat = output.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def run_language_model_training(model, train_loader, val_loader, vocab_size, epochs=10, lr=0.001):
    """Main training loop for language models."""
    print(f"\n🧠 Training {model.__class__.__name__}...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, vocab_size)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, vocab_size)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), f'{model.__class__.__name__}_best.pt')

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {val_ppl:7.3f}")

    print(f"✅ {model.__class__.__name__} training complete!")
    # model.load_state_dict(torch.load(f'{model.__class__.__name__}_best.pt'))
    return model, {'loss': best_val_loss, 'perplexity': math.exp(best_val_loss)}
