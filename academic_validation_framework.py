#!/usr/bin/env python3
"""
Research Validation Framework for Learned Encoding Paper
This framework runs REAL experiments to test:

1. Compression scaling
2. Vocabulary scaling
3. Architecture robustness
"""

import json
import time
import random
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import LearnedEncodingModel, Autoencoder, AutoencoderBaselineModel
from data_loader import get_processed_data, get_batch

DEVICE = "cpu"

@dataclass
class ValidationConfig:
    compression_ratios: List[float]
    vocabulary_sizes: List[int]
    model_architectures: List[Dict]
    num_independent_runs: int = 3
    epochs: int = 1
    bptt: int = 35
    batch_size: int = 20

def train(model: nn.Module, train_data: torch.Tensor, bptt: int, criterion, optimizer, epoch: int):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        # The model expects [batch_size, seq_len], but get_batch returns [seq_len, batch_size]
        # So we transpose it.
        output = model(data.t())
        loss = criterion(output.view(-1, output.size(-1)), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % 50 == 0 and batch > 0:
            cur_loss = total_loss / 50
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} | {batch:5d}/{train_data.size(0) // bptt:5d} batches | '
                  f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: torch.Tensor, bptt: int, criterion):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            output = model(data.t())
            output_flat = output.view(-1, output.size(-1))
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

class AcademicValidationFramework:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = {
            'compression_scaling': {},
            'vocabulary_scaling': {},
            'architecture_ablations': {},
        }
        print("🎓 Academic Validation Framework Initialized for REAL Experiments")

    def _run_real_experiment(self, experiment_params: Dict) -> Dict:
        print(f"    Running experiment with params: {experiment_params}")
        
        torch.manual_seed(experiment_params['seed'])
        
        train_data, val_data, _, vocab = get_processed_data(
            self.config.batch_size, DEVICE, experiment_params['vocab_size']
        )
        
        vocab_size = len(vocab)
        model_config = experiment_params['architecture']
        
        if experiment_params['approach'] == 'learned':
            model = LearnedEncodingModel(
                vocab_size, experiment_params['embedding_dim'], 2,
                model_config['layers'], model_config['hidden_dim']
            ).to(DEVICE)
        else: # autoencoder
            original_dim = experiment_params['original_dim']
            compressed_dim = experiment_params['embedding_dim']
            autoencoder = Autoencoder(original_dim, compressed_dim).to(DEVICE)
            model = AutoencoderBaselineModel(
                vocab_size, original_dim, compressed_dim, 2,
                model_config['layers'], model_config['hidden_dim'], autoencoder
            ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            train(model, train_data, self.config.bptt, criterion, optimizer, epoch)
            val_loss = evaluate(model, val_data, self.config.bptt, criterion)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | '
                  f'valid ppl {math.exp(val_loss):8.2f}')
            print('-' * 89)

        return {'perplexity': math.exp(val_loss)}

    def run_compression_scaling_study(self, vocab_size=5000, original_dim=128):
        print("\n📐 Compression Scaling Study")
        
        for ratio in self.config.compression_ratios:
            print(f"\n🔬 Testing {ratio:.1f}:1 compression")
            compressed_dim = int(original_dim / ratio)
            
            learned_results, traditional_results = [], []
            
            for run in range(self.config.num_independent_runs):
                print(f"  Run {run + 1}/{self.config.num_independent_runs}...")
                seed = 42 + run
                arch = self.config.model_architectures[0]

                res_learned = self._run_real_experiment({
                    'approach': 'learned', 'seed': seed, 'vocab_size': vocab_size,
                    'embedding_dim': compressed_dim, 'architecture': arch
                })
                learned_results.append(res_learned['perplexity'])
                
                res_ae = self._run_real_experiment({
                    'approach': 'autoencoder', 'seed': seed, 'vocab_size': vocab_size,
                    'embedding_dim': compressed_dim, 'original_dim': original_dim, 'architecture': arch
                })
                traditional_results.append(res_ae['perplexity'])

            stats = self._compute_statistical_metrics(learned_results, traditional_results)
            self.results['compression_scaling'][ratio] = stats
            print(f"  📊 Learned PPL: {stats['learned_mean']:.2f} ± {stats['learned_ci']:.2f}")
            print(f"  📊 Autoencoder PPL: {stats['traditional_mean']:.2f} ± {stats['traditional_ci']:.2f}")

    def _compute_statistical_metrics(self, learned_results: List[float], traditional_results: List[float]) -> Dict:
        learned_mean = statistics.mean(learned_results)
        learned_std = statistics.stdev(learned_results) if len(learned_results) > 1 else 0.0
        traditional_mean = statistics.mean(traditional_results)
        traditional_std = statistics.stdev(traditional_results) if len(traditional_results) > 1 else 0.0
        
        n = len(learned_results)
        if n > 1:
            t_critical = 3.182 if n == 3 else (4.303 if n==2 else 1.96) # t-dist critical value
            learned_ci = t_critical * learned_std / math.sqrt(n)
            traditional_ci = t_critical * traditional_std / math.sqrt(n)
        else:
            learned_ci = 0.0
            traditional_ci = 0.0

        return {
            'learned_mean': learned_mean, 'learned_std': learned_std, 'learned_ci': learned_ci,
            'traditional_mean': traditional_mean, 'traditional_std': traditional_std, 'traditional_ci': traditional_ci,
            'sample_size': n
        }

    def save_comprehensive_results(self, filename: str = "real_experiment_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Comprehensive validation results saved to {filename}")

def main():
    config = ValidationConfig(
        compression_ratios=[4.0, 8.0],
        vocabulary_sizes=[5000],
        model_architectures=[{'layers': 2, 'hidden_dim': 128, 'name': 'small'}],
        num_independent_runs=3,
        epochs=3
    )
    validator = AcademicValidationFramework(config)
    validator.run_compression_scaling_study()
    validator.save_comprehensive_results()
    print(f"\n🏁 Academic Validation Complete!")

if __name__ == "__main__":
    main()
