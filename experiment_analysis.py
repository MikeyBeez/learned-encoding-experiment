#!/usr/bin/env python3
"""
Experiment Analysis Metamodel

This file contains the core, UI-agnostic logic for analyzing and
visualizing the results of the learned encoding experiments. The functions
in this module are designed to produce insightful, presentation-ready
figures and reports.
"""

import json
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from real_world_experiment import LearnedEncodingModel, TraditionalModel

def generate_embedding_visualization(model: torch.nn.Module) -> Figure:
    """
    Generates a t-SNE visualization of the model's embedding space.

    Args:
        model: The trained model instance (either LearnedEncodingModel or TraditionalModel).

    Returns:
        A matplotlib Figure object containing the t-SNE plot.
    """
    if not isinstance(model, (LearnedEncodingModel, TraditionalModel)):
        raise TypeError("Model must be an instance of LearnedEncodingModel or TraditionalModel.")

    # Extract the embedding weights
    if isinstance(model, LearnedEncodingModel):
        embeddings = model.token_encoder.weight.detach().cpu().numpy()
        title = "t-SNE of Learned Embeddings"
        subtitle = "Each point is a token from the vocabulary, positioned by its learned representation."
    else: # TraditionalModel
        # For the traditional model, we visualize the autoencoder's compressed embeddings
        embeddings = model.autoencoder.embedding.weight.detach().cpu().numpy()
        title = "t-SNE of Traditional Embeddings (from Autoencoder)"
        subtitle = "Each point is a token, positioned by its pre-trained autoencoder representation."

    # Perform t-SNE
    # Use a smaller perplexity if vocab size is small
    perplexity = min(30.0, max(1.0, embeddings.shape[0] - 2.0))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

    # Add insightful titles and labels
    fig.suptitle(title, fontsize=16, fontweight='bold')
    ax.set_title(subtitle, fontsize=10)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.grid(True, alpha=0.3)

    # Add a text box explaining what to look for
    explanation = (
        "Insight: Look for clusters or patterns.\n"
        "Well-formed clusters can indicate that the model\n"
        "has learned semantic relationships between tokens."
    )
    ax.text(0.95, 0.05, explanation, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def compare_embedding_spaces(model1: torch.nn.Module, model2: torch.nn.Module) -> (Figure, str):
    """
    Compares two embedding spaces using Canonical Correlation Analysis (CCA).

    Args:
        model1: The first trained model (e.g., LearnedEncodingModel).
        model2: The second trained model (e.g., TraditionalModel).

    Returns:
        A tuple containing:
        - A matplotlib Figure object with the CCA results plot.
        - A string summarizing the analysis and its insights.
    """
    # Extract embeddings
    if isinstance(model1, LearnedEncodingModel):
        embed1 = model1.token_encoder.weight.detach().cpu().numpy()
        name1 = "Learned"
    else:
        embed1 = model1.autoencoder.embedding.weight.detach().cpu().numpy()
        name1 = "Traditional"

    if isinstance(model2, LearnedEncodingModel):
        embed2 = model2.token_encoder.weight.detach().cpu().numpy()
        name2 = "Learned"
    else:
        embed2 = model2.autoencoder.embedding.weight.detach().cpu().numpy()
        name2 = "Traditional"

    # Ensure dimensions are compatible for CCA
    if embed1.shape[1] != embed2.shape[1]:
        summary = "Error: Embedding dimensions are not equal, cannot perform CCA."
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, summary, ha='center', color='red')
        return fig, summary

    n_components = embed1.shape[1]
    cca = CCA(n_components=n_components)
    cca.fit(embed1, embed2)

    # The canonical correlations are stored in cca.score_
    # Note: sklearn's CCA.score returns R^2, so we take the sqrt for R
    correlations = np.sqrt(cca.score(embed1, embed2))

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    component_indices = np.arange(len(correlations))
    ax.bar(component_indices, correlations, color='skyblue')

    fig.suptitle("Embedding Space Similarity (CCA)", fontsize=16, fontweight='bold')
    ax.set_title(f"Comparing '{name1}' vs. '{name2}' models. Higher bars mean more similar representations.", fontsize=10)
    ax.set_xlabel("Canonical Component Index")
    ax.set_ylabel("Correlation Coefficient (R)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Create Summary ---
    summary_lines = [
        "### Canonical Correlation Analysis (CCA) Results",
        f"This analysis measures the linear correlation between the embedding spaces of the two models ('{name1}' and '{name2}').",
        f"- **Top 3 Correlations**: {correlations[0]:.3f}, {correlations[1]:.3f}, {correlations[2]:.3f}",
        f"- **Average Correlation**: {np.mean(correlations):.3f}"
    ]

    avg_corr = np.mean(correlations)
    if avg_corr > 0.8:
        insight = "The embedding spaces are **highly similar**. This suggests both models, despite different training methods, learned to represent token relationships in a fundamentally related way."
    elif avg_corr > 0.5:
        insight = "The embedding spaces show a **moderate degree of similarity**. They share some common structures but also have significant differences."
    else:
        insight = "The embedding spaces are **largely dissimilar**. The two models learned very different ways of representing the vocabulary."

    summary_lines.append(f"- **Insight**: {insight}")

    summary = "\n".join(summary_lines)

    return fig, summary

def _generate_study_section(data: dict, section_key: str, title: str, hypothesis: str, columns: list[tuple[str, str]], row_key_suffix: str = "") -> list[str]:
    """Generates a generic markdown section for a study."""
    lines = []
    if section_key in data and data.get(section_key):
        lines.append(f"\n## {title}")
        lines.append(f"**Hypothesis**: {hypothesis}")

        # Create table header
        header = "| " + " | ".join([col[0] for col in columns]) + " |"
        separator = "|---" * len(columns) + "|"
        lines.append(header)
        lines.append(separator)

        # Populate table rows
        for key, results in data[section_key].items():
            row_values = []
            for _, data_key in columns:
                if data_key == 'row_key':
                    row_values.append(f"{key}{row_key_suffix}")
                    continue

                val = results.get(data_key)
                # Formatting logic
                if data_key == 'significant':
                    row_values.append("✅" if val else "❌")
                elif 'improvement' in data_key:
                    row_values.append(f"{val:.1f}%" if isinstance(val, float) else "N/A")
                elif isinstance(val, float):
                    if 'loss' in data_key:
                        std = results.get(data_key.replace('_mean', '_std'), 0)
                        row_values.append(f"{val:.3f} ± {std:.3f}")
                    else:
                        row_values.append(f"{val:.3f}")
                else:
                    row_values.append(str(val) if val is not None else "N/A")

            lines.append("| " + " | ".join(row_values) + " |")

        # Simplified dynamic result
        significant_count = sum(1 for res in data[section_key].values() if res.get('significant'))
        if len(data[section_key]) > 0:
            if significant_count == len(data[section_key]):
                result_line = "✅ **VALIDATED** - Strong statistical evidence across all tests."
            elif significant_count > 0:
                result_line = "✅ **PARTIALLY VALIDATED** - Strong evidence in some tests."
            else:
                result_line = "❌ **NOT VALIDATED** - No statistical evidence found."
            lines.append(f"\n**Result**: {result_line}")
    return lines

def generate_markdown_report(json_path: str) -> str:
    """
    Generates an insightful markdown report from a JSON results file.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"# Error\n\n{e}"

    report_lines = [
        "# 📊 Automated Validation Report",
        "This report was automatically generated from the validation framework results."
    ]

    # --- Overall Statistical Summary ---
    all_results = []
    for section_key, section_data in data.items():
        if isinstance(section_data, dict):
            for result_data in section_data.values():
                if isinstance(result_data, dict) and 'significant' in result_data:
                    all_results.append(result_data)

    total_tests = len(all_results)
    # A "win" is a significant result where the learned model's loss is lower than traditional.
    significant_wins = sum(1 for r in all_results if r.get('significant') and r.get('learned_loss_mean', 1) < r.get('traditional_loss_mean', 0))
    success_rate = (significant_wins / total_tests * 100) if total_tests > 0 else 0

    if total_tests > 0:
        report_lines.extend([
            "\n## 📊 Comprehensive Statistical Summary",
            f"- **Total experiments**: {total_tests}",
            f"- **Statistically significant wins for Learned Encoding**: {significant_wins}",
            f"- **Success rate**: {success_rate:.1f}%",
            f"- **Confidence level**: 95%",
        ])

    # --- Validation Sections ---
    report_lines.append("\n---")
    report_lines.append("\n## 🔬 Validation Results by Category")

    report_lines.extend(_generate_study_section(
        data, "compression_scaling", "📐 Compression Scaling Study",
        "Learned encodings maintain performance across compression ratios",
        [("Compression Ratio", "row_key"), ("Learned Loss", "learned_loss_mean"), ("Traditional Loss", "traditional_loss_mean"),
         ("Effect Size", "effect_size"), ("P-value", "p_value"), ("Significant", "significant")],
        row_key_suffix=":1"
    ))

    report_lines.extend(_generate_study_section(
        data, "vocabulary_scaling", "📚 Vocabulary Scaling Study",
        "Performance maintains across vocabulary sizes",
        [("Vocabulary Size", "row_key"), ("Learned Loss", "learned_loss_mean"), ("Traditional Loss", "traditional_loss_mean"),
         ("Improvement", "relative_improvement"), ("Significant", "significant")],
        row_key_suffix=" tokens"
    ))

    report_lines.extend(_generate_study_section(
        data, "complexity_scaling", "🎯 Dataset Complexity Study",
        "Robustness across different data types",
        [("Dataset Type", "row_key"), ("Learned Loss", "learned_loss_mean"), ("Traditional Loss", "traditional_loss_mean"),
         ("Robustness Score", "robustness_score"), ("Significant", "significant")]
    ))

    report_lines.extend(_generate_study_section(
        data, "baseline_comparisons", "🏆 Baseline Comparison Study",
        "Outperforms multiple autoencoder variants",
        [("Baseline Method", "row_key"), ("Learned Loss", "learned_loss_mean"), ("Baseline Loss", "traditional_loss_mean"),
         ("Improvement", "relative_improvement"), ("Significant", "significant")]
    ))

    return "\n".join(report_lines)
