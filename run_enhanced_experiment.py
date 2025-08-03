#!/usr/bin/env python3
"""
Main entry point for running the enhanced Learned Encoding Experiment.

This script orchestrates the full, rigorous experimental pipeline:
1. Configures the experiment, including parametric sweeps.
2. Runs the experiment using the research-grade `academic_validation_framework`.
3. Generates a detailed, human-readable report of the findings.
4. Saves the raw results to a JSON file for further analysis.
"""

import json
from academic_validation_framework import ExperimentConfig, ExperimentRunner
from reporting import generate_report

def main():
    """
    Configure, run, and report on the learned encoding experiment.
    """
    print("🚀 Kicking off the Enhanced Learned Encoding Experiment 🚀")
    print("="*60)

    # 1. Configure the experiment
    # We can easily change parameters here for different studies.
    # For this run, we'll test three different compression ratios.
    # A more representative config that is still fast enough for CI/testing
    config = ExperimentConfig(
        num_runs=2,
        learned_dims=[8, 4], # Test 4:1 and 8:1 compression
        epochs=15,
        autoencoder_epochs=10
    )

    print("📝 Configuration:")
    print(f"  - Vocab Size: {config.vocab_size}")
    print(f"  - Traditional Dim: {config.traditional_dim}")
    print(f"  - Learned Dims (sweep): {config.learned_dims}")
    print(f"  - Num. Statistical Runs: {config.num_runs}")
    print(f"  - Epochs: {config.epochs}")
    print("="*60)

    # 2. Run the experiment
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()

    # 3. Generate and print the report
    report = generate_report(results)
    print("\n\n" + "="*60)
    print(report)
    print("="*60)

    # 4. Save detailed results to a file
    results_filename = "enhanced_experiment_results.json"
    # The config is already a dict within the results, so no conversion needed
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Full results saved to `{results_filename}`")
    print("🏁 Enhanced Experiment Finished! 🏁")

if __name__ == "__main__":
    main()
