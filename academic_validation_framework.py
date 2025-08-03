#!/usr/bin/env python3
"""
A research-grade, modular neural network framework for the Learned Encoding Experiment.

This framework replaces the original, flawed implementation with a correct,
extensible, and understandable architecture based on modern deep learning principles.
It features a layer-based design, proper backpropagation, and a clear separation
of concerns between models, layers, and optimizers.
"""

import random
import math
from typing import List, Dict, Any
from dataclasses import dataclass, field

# =============================================================================
# 1. Core Matrix and Optimizer Classes
# =============================================================================

class Matrix:
    """A simple matrix class that supports data and gradient storage."""
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # Use Xavier/Glorot initialization for better weight stability
        limit = math.sqrt(6.0 / (rows + cols))
        self.data = [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
        self.grad = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def __repr__(self):
        return f"Matrix(rows={self.rows}, cols={self.cols})"

    def zero_grad(self):
        """Resets the gradients to zero."""
        self.grad = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]

class SGD:
    """A simple Stochastic Gradient Descent optimizer."""
    def __init__(self, params: List[Matrix], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self):
        """Updates parameters using their stored gradients."""
        for param in self.params:
            for i in range(param.rows):
                for j in range(param.cols):
                    param.data[i][j] -= self.lr * param.grad[i][j]

    def zero_grad(self):
        """Resets gradients for all managed parameters."""
        for param in self.params:
            param.zero_grad()

# =============================================================================
# 2. Modular Layer-Based Architecture
# =============================================================================

class Layer:
    """Abstract base class for a neural network layer."""
    def forward(self, inputs: Any) -> Any:
        raise NotImplementedError

    def backward(self, grad_outputs: Any) -> Any:
        raise NotImplementedError

    def get_params(self) -> List[Matrix]:
        """Returns the trainable parameters of the layer."""
        return []

    @property
    def output_dim(self) -> int:
        """Returns the output dimension of the layer."""
        raise NotImplementedError

class EmbeddingLayer(Layer):
    """
    An embedding layer that maps token IDs to dense vectors.
    This is the core component of the experiment.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = Matrix(vocab_size, embed_dim)
        self.inputs = None

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(self, inputs: List[int]) -> List[List[float]]:
        """Maps a list of token IDs to a list of embedding vectors."""
        self.inputs = inputs
        return [self.embeddings.data[token_id] for token_id in inputs]

    def backward(self, grad_outputs: List[List[float]]):
        """
        Calculates gradients for the embedding matrix.
        The gradient for a specific token's embedding is the sum of all
        gradients that flowed back to it.
        """
        # No gradient flows "back" from an embedding layer.
        # It's the start of the chain for this experiment's purpose.
        for i, token_id in enumerate(self.inputs):
            for j in range(self.embed_dim):
                self.embeddings.grad[token_id][j] += grad_outputs[i][j]
        return None

    def get_params(self) -> List[Matrix]:
        return [self.embeddings]

class LinearLayer(Layer):
    """A standard linear transformation layer (fully connected)."""
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = Matrix(input_dim, output_dim)
        self.biases = Matrix(1, output_dim)
        self.inputs = None

    @property
    def output_dim(self) -> int:
        return self.weights.cols

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Performs the forward pass: output = input @ weights + biases."""
        self.inputs = inputs
        batch_size = len(inputs)
        outputs = [[0.0] * self.weights.cols for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(self.weights.cols):
                dot_product = 0.0
                for k in range(self.weights.rows):
                    dot_product += inputs[i][k] * self.weights.data[k][j]
                outputs[i][j] = dot_product + self.biases.data[0][j]
        return outputs

    def backward(self, grad_outputs: List[List[float]]):
        """Performs the backward pass to compute gradients."""
        batch_size = len(self.inputs)
        grad_inputs = [[0.0] * self.weights.rows for _ in range(batch_size)]

        # Calculate gradients for weights, biases, and inputs
        for i in range(batch_size):
            for j in range(self.weights.cols):
                grad_output_val = grad_outputs[i][j]
                # Gradient for bias
                self.biases.grad[0][j] += grad_output_val
                for k in range(self.weights.rows):
                    # Gradient for weight
                    self.weights.grad[k][j] += self.inputs[i][k] * grad_output_val
                    # Gradient for input to this layer
                    grad_inputs[i][k] += self.weights.data[k][j] * grad_output_val
        return grad_inputs

    def get_params(self) -> List[Matrix]:
        return [self.weights, self.biases]

class ReLULayer(Layer):
    """A Rectified Linear Unit (ReLU) activation layer."""
    def __init__(self):
        self.inputs = None
        self._output_dim = 0

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Applies the ReLU function element-wise."""
        self.inputs = inputs
        if inputs:
            self._output_dim = len(inputs[0])
        return [[max(0.0, x) for x in row] for row in inputs]

    def backward(self, grad_outputs: List[List[float]]):
        """Computes the gradient through the ReLU activation."""
        grad_inputs = [[0.0] * len(row) for row in self.inputs]
        for i in range(len(self.inputs)):
            for j in range(len(self.inputs[i])):
                # Gradient is 1 if input was > 0, else 0
                if self.inputs[i][j] > 0:
                    grad_inputs[i][j] = grad_outputs[i][j]
        return grad_inputs

# =============================================================================
# 3. Model and Loss Function Classes
# =============================================================================

class Sequential(Layer):
    """A container for a sequence of layers."""
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, inputs: Any) -> Any:
        """Performs a forward pass through all layers in sequence."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_outputs: Any):
        """Performs a backward pass through all layers in reverse."""
        for layer in reversed(self.layers):
            grad_outputs = layer.backward(grad_outputs)
        return grad_outputs

    def get_params(self) -> List[Matrix]:
        """Gathers parameters from all contained layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

class CrossEntropyLoss:
    """
    Computes the cross-entropy loss and its gradient.
    This implementation combines softmax and negative log likelihood.
    """
    def __init__(self):
        self.logits = None
        self.targets = None

    def forward(self, logits: List[List[float]], targets: List[int]) -> float:
        """Calculates the average cross-entropy loss."""
        self.logits = logits
        self.targets = targets
        batch_size = len(logits)
        total_loss = 0.0

        for i in range(batch_size):
            # Softmax calculation
            row_logits = logits[i]
            max_logit = max(row_logits)
            exp_logits = [math.exp(l - max_logit) for l in row_logits]
            sum_exp_logits = sum(exp_logits)
            probs = [e / sum_exp_logits for e in exp_logits]

            # Negative log likelihood
            target_prob = probs[targets[i]]
            total_loss += -math.log(target_prob + 1e-9) # Add epsilon for stability

        return total_loss / batch_size

    def backward(self) -> List[List[float]]:
        """Computes the gradient of the loss with respect to the logits."""
        batch_size = len(self.logits)
        vocab_size = len(self.logits[0])
        grad_logits = [[0.0] * vocab_size for _ in range(batch_size)]

        for i in range(batch_size):
            # Softmax calculation (same as forward)
            row_logits = self.logits[i]
            max_logit = max(row_logits)
            exp_logits = [math.exp(l - max_logit) for l in row_logits]
            sum_exp_logits = sum(exp_logits)
            probs = [e / sum_exp_logits for e in exp_logits]

            # Gradient calculation (y_hat - y)
            for j in range(vocab_size):
                indicator = 1.0 if j == self.targets[i] else 0.0
                grad_logits[i][j] = (probs[j] - indicator) / batch_size

        return grad_logits

class MSELoss:
    """
    Computes the Mean Squared Error loss and its gradient.
    Used for the autoencoder's reconstruction task.
    """
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions: List[List[float]], targets: List[List[float]]) -> float:
        """Calculates the average Mean Squared Error."""
        self.predictions = predictions
        self.targets = targets
        batch_size = len(predictions)
        num_dims = len(predictions[0])
        total_error = 0.0

        for i in range(batch_size):
            for j in range(num_dims):
                error = predictions[i][j] - targets[i][j]
                total_error += error * error

        return total_error / (batch_size * num_dims)

    def backward(self) -> List[List[float]]:
        """Computes the gradient of the MSE loss."""
        batch_size = len(self.predictions)
        num_dims = len(self.predictions[0])

        grad_inputs = [[0.0] * num_dims for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(num_dims):
                error = self.predictions[i][j] - self.targets[i][j]
                grad_inputs[i][j] = 2 * error / (batch_size * num_dims)

        return grad_inputs

# =============================================================================
# 4. Model Architectures
# =============================================================================

def build_autoencoder(input_dim: int, compressed_dim: int) -> Sequential:
    """Factory function to build a simple autoencoder model."""
    return Sequential([
        LinearLayer(input_dim, compressed_dim),
        ReLULayer(),
        LinearLayer(compressed_dim, input_dim)
    ])

class PretrainedEncoderLayer(Layer):
    """A layer that uses a pretrained, frozen encoder."""
    def __init__(self, full_embeddings: Matrix, encoder: Layer):
        self.full_embeddings = full_embeddings
        self.encoder = encoder
        self.inputs = None

    @property
    def output_dim(self) -> int:
        # The output dimension is the output dimension of the encoder itself.
        return self.encoder.output_dim

    def forward(self, inputs: List[int]) -> List[List[float]]:
        """
        Takes token IDs, gets their full embeddings, then encodes them.
        The encoder is a LinearLayer, so its forward method expects a list of lists.
        """
        self.inputs = inputs
        # 1. Get the high-dimensional embeddings for the input tokens
        original_embeds = [self.full_embeddings.data[token_id] for token_id in inputs]
        # 2. Pass these embeddings through the pretrained encoder
        compressed_embeds = self.encoder.forward(original_embeds)
        return compressed_embeds

    def backward(self, grad_outputs: Any):
        # Gradients are not propagated back through the frozen encoder or original embeddings.
        # This layer is a starting point for the downstream model.
        return None

    def get_params(self) -> List[Matrix]:
        # This layer has no trainable parameters.
        return []

def build_downstream_model(embedding_layer: Layer, hidden_dim: int, vocab_size: int) -> Sequential:
    """
    Factory function to build the downstream model for next-token prediction.
    It accepts an embedding layer as a parameter to facilitate both the
    'learned' and 'traditional' experimental setups with the exact same
    downstream architecture, ensuring a fair and rigorous comparison.
    """
    embed_dim = embedding_layer.output_dim
    return Sequential([
        embedding_layer,
        LinearLayer(embed_dim, hidden_dim),
        ReLULayer(),
        LinearLayer(hidden_dim, vocab_size)
    ])

def main():
    print("Academic Validation Framework: All components are ready.")
    print("This file provides a complete, research-grade toolkit for building and training models.")
    print("Next steps: Implement the ExperimentRunner to conduct the validation study.")

# =============================================================================
# 5. Rigorous Experiment Runner
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the validation experiment."""
    vocab_size: int = 20
    traditional_dim: int = 32
    hidden_dim: int = 32
    seq_len: int = 8
    num_sequences: int = 200
    epochs: int = 25
    autoencoder_epochs: int = 15
    learning_rate: float = 0.1
    num_runs: int = 5 # Reduced for speed in typical runs
    # Parametric sweep configuration
    learned_dims: List[int] = field(default_factory=lambda: [4, 8, 16])

class ExperimentRunner:
    """
    Manages the full, rigorous experiment comparing learned vs. traditional embeddings.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.training_data = self._generate_training_data()
        # Create a set of high-dimensional "ground truth" embeddings to be compressed
        self.ground_truth_embeddings = Matrix(config.vocab_size, config.traditional_dim)

    def _generate_training_data(self) -> List[Dict[str, List[int]]]:
        """Generates synthetic, pattern-based training data."""
        patterns = [
            [1, 2, 3, 4, 5] * 3, [7, 8, 9] * 4,
            [11, 12, 13, 14] * 3, [16, 17] * 6,
        ]
        sequences = []
        for _ in range(self.config.num_sequences):
            pattern = random.choice(patterns)
            start_idx = random.randint(0, max(0, len(pattern) - self.config.seq_len))
            sequence = pattern[start_idx : start_idx + self.config.seq_len]
            while len(sequence) < self.config.seq_len:
                sequence.extend(pattern)
            sequence = sequence[:self.config.seq_len]
            sequences.append({
                "inputs": sequence[:-1],
                "targets": sequence[1:]
            })
        return sequences

    def _train_autoencoder(self, learned_dim: int) -> Layer:
        """
        Phase 1: Train an autoencoder to compress the ground truth embeddings.
        Returns the trained encoder layer.
        """
        print(f"  Training autoencoder ({self.config.traditional_dim}D -> {learned_dim}D)...")
        autoencoder = build_autoencoder(self.config.traditional_dim, learned_dim)
        optimizer = SGD(autoencoder.get_params(), lr=self.config.learning_rate)
        loss_fn = MSELoss()

        all_embeddings = self.ground_truth_embeddings.data

        for epoch in range(self.config.autoencoder_epochs):
            optimizer.zero_grad()
            reconstructed = autoencoder.forward(all_embeddings)
            loss = loss_fn.forward(reconstructed, all_embeddings)
            grad = loss_fn.backward()
            autoencoder.backward(grad)
            optimizer.step()

        print("  Autoencoder training complete.")
        return autoencoder.layers[0]

    def _run_single_comparative_trial(self, learned_dim: int, pretrained_encoder: Layer) -> Dict[str, float]:
        """Phase 2: Run one trial of the learned vs. traditional comparison."""
        loss_fn = CrossEntropyLoss()

        # 1. Learned Encoding Model
        learned_embedding_layer = EmbeddingLayer(self.config.vocab_size, learned_dim)
        learned_model = build_downstream_model(learned_embedding_layer, self.config.hidden_dim, self.config.vocab_size)
        learned_optimizer = SGD(learned_model.get_params(), lr=self.config.learning_rate)

        # 2. Traditional Model
        traditional_embedding_layer = PretrainedEncoderLayer(self.ground_truth_embeddings, pretrained_encoder)
        traditional_model = build_downstream_model(traditional_embedding_layer, self.config.hidden_dim, self.config.vocab_size)
        traditional_params = [p for p in traditional_model.get_params() if p not in traditional_embedding_layer.get_params()]
        traditional_optimizer = SGD(traditional_params, lr=self.config.learning_rate)

        # 3. Naive Baseline Model
        naive_embedding_layer = EmbeddingLayer(self.config.vocab_size, learned_dim)
        naive_model = build_downstream_model(naive_embedding_layer, self.config.hidden_dim, self.config.vocab_size)
        naive_params = [p for p in naive_model.get_params() if p not in naive_embedding_layer.get_params()]
        naive_optimizer = SGD(naive_params, lr=self.config.learning_rate)

        # Training loop
        for _ in range(self.config.epochs):
            for batch in self.training_data:
                # Train all three models
                for model, optimizer in [(learned_model, learned_optimizer), (traditional_model, traditional_optimizer), (naive_model, naive_optimizer)]:
                    optimizer.zero_grad()
                    logits = model.forward(batch["inputs"])
                    loss = loss_fn.forward(logits, batch["targets"])
                    grad = loss_fn.backward()
                    model.backward(grad)
                    optimizer.step()

        # Evaluate final losses
        final_losses = {}
        for name, model in [("learned", learned_model), ("traditional", traditional_model), ("naive", naive_model)]:
            loss = sum(loss_fn.forward(model.forward(b["inputs"]), b["targets"]) for b in self.training_data) / len(self.training_data)
            final_losses[name] = loss

        return final_losses

    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Runs the entire multi-run, multi-parameter experiment.
        """
        full_results = {"config": self.config.__dict__, "results_by_ratio": {}}

        for learned_dim in self.config.learned_dims:
            compression_ratio = self.config.traditional_dim / learned_dim
            print(f"🚀 Starting experiment for compression ratio: {compression_ratio:.1f}:1 ({learned_dim}D)")
            
            run_losses = {"learned": [], "traditional": [], "naive": []}

            for i in range(self.config.num_runs):
                print(f"\n--- Run {i+1}/{self.config.num_runs} for {learned_dim}D ---")
                random.seed(42 + i)

                encoder = self._train_autoencoder(learned_dim)

                print("  Running comparative trial...")
                final_losses = self._run_single_comparative_trial(learned_dim, encoder)
                run_losses["learned"].append(final_losses["learned"])
                run_losses["traditional"].append(final_losses["traditional"])
                run_losses["naive"].append(final_losses["naive"])
                print(f"  Trial complete. Losses -> "
                      f"Learned: {final_losses['learned']:.4f}, "
                      f"Traditional: {final_losses['traditional']:.4f}, "
                      f"Naive: {final_losses['naive']:.4f}")

            # Calculate statistics for this compression ratio
            def get_stats(data: List[float]) -> Dict[str, float]:
                mean = sum(data) / len(data)
                std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / (len(data) - 1)) if len(data) > 1 else 0.0
                return {"mean_loss": mean, "std_dev": std_dev}

            full_results["results_by_ratio"][f"{compression_ratio:.0f}:1"] = {
                "learned_model": get_stats(run_losses["learned"]),
                "traditional_model": get_stats(run_losses["traditional"]),
                "naive_model": get_stats(run_losses["naive"])
            }
        
        print("\n🏁 Experiment Complete")
        return full_results

if __name__ == "__main__":
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()

    # Basic print of final results
    import json
    print("\n--- Final Results Summary ---")
    print(json.dumps(results, indent=2))
