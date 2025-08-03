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

def generate_markdown_report(json_path: str) -> str:
    """
    Generates an insightful markdown report from a JSON results file.

    Args:
        json_path: Path to the JSON file containing validation results.

    Returns:
        A string containing the formatted markdown report.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"# Error\n\nCould not find file: `{json_path}`"
    except json.JSONDecodeError:
        return f"# Error\n\nCould not decode invalid JSON from file: `{json_path}`"

    report_lines = [
        "# 📊 Automated Validation Report",
        "This report was automatically generated from the validation framework results."
    ]

    # --- Compression Scaling Section ---
    if 'compression_scaling' in data and data['compression_scaling']:
        report_lines.append("\n## 📐 Compression Scaling Study")
        report_lines.append("This study evaluates the performance of the Learned vs. Traditional (Autoencoder) model across different embedding compression ratios.")

        # Create table header
        header = "| Compression Ratio | Learned PPL | Traditional PPL | Advantage | Statistically Significant? |"
        separator = "|---|---|---|---|---|"
        report_lines.append(header)
        report_lines.append(separator)

        # Populate table rows
        for ratio, results in data['compression_scaling'].items():
            learned_ppl = results.get('learned_perplexity_mean', 'N/A')
            trad_ppl = results.get('traditional_perplexity_mean', 'N/A')

            # Format numbers if they exist
            if isinstance(learned_ppl, (int, float)):
                learned_ppl_str = f"{learned_ppl:.2f}"
            else:
                learned_ppl_str = "N/A"

            if isinstance(trad_ppl, (int, float)):
                trad_ppl_str = f"{trad_ppl:.2f}"
            else:
                trad_ppl_str = "N/A"

            # Determine advantage
            advantage = "None"
            if isinstance(learned_ppl, (int, float)) and isinstance(trad_ppl, (int, float)):
                if learned_ppl < trad_ppl:
                    improvement = ((trad_ppl - learned_ppl) / trad_ppl) * 100
                    advantage = f"**Learned (+{improvement:.1f}%)**"
                else:
                    improvement = ((learned_ppl - trad_ppl) / learned_ppl) * 100
                    advantage = f"Traditional (+{improvement:.1f}%)"

            significant_str = "✅ Yes" if results.get('significant', False) else "❌ No"

            row = f"| {ratio}:1 | {learned_ppl_str} | {trad_ppl_str} | {advantage} | {significant_str} |"
            report_lines.append(row)

        # --- Insight Summary ---
        report_lines.append("\n### Key Insights")
        significant_wins = [r for r, res in data['compression_scaling'].items() if res.get('significant', False) and res.get('learned_perplexity_mean', float('inf')) < res.get('traditional_perplexity_mean', 0)]

        if len(significant_wins) == len(data['compression_scaling']):
            report_lines.append("- **Consistent Advantage**: The **Learned Encoding** model shows a statistically significant advantage across **all tested compression ratios**.")
            report_lines.append("- **Recommendation**: This provides strong evidence that for this task, learning embeddings as part of the main training objective is superior to the pre-trained autoencoder approach.")
        elif significant_wins:
            report_lines.append(f"- **Situational Advantage**: The **Learned Encoding** model showed a statistically significant advantage at the following compression ratios: **{', '.join(significant_wins)}:1**.")
            report_lines.append("- **Recommendation**: The learned approach is highly advantageous in these specific scenarios. Further testing may be needed for other ratios.")
        else:
            report_lines.append("- **No Clear Advantage**: Neither model showed a consistent, statistically significant advantage.")
            report_lines.append("- **Recommendation**: Both models perform similarly. The choice may depend on other factors like training time or implementation complexity.")

    else:
        report_lines.append("\nNo compression scaling results found in the file.")

    return "\n".join(report_lines)
