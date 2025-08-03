#!/usr/bin/env python3
"""
Quick test run of the learned encoding experiment
"""

import sys
import subprocess

def main():
    """Run the experiment with dependencies check."""
    print("🔬 Learned Encoding Experiment - Environment Setup")
    print("=" * 50)
    
    # Check if we have required packages for real-world validation
    try:
        import torch
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import datasets
        print("✅ Core dependencies found: torch, pandas, numpy, scikit-learn, matplotlib, datasets")
    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        print("Installing required packages for real-world validation...")
        # Install torch for CPU first to save space in constrained environments
        print("Installing PyTorch (CPU version)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
        print("Installing other dependencies...")
        packages = ["pandas", "numpy", "scikit-learn", "matplotlib", "datasets"]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ Core dependencies installed!")

    print("\n✅ Environment is set up for real-world validation.")
    print("💡 To run the new validation suite, execute:")
    print("   python3 academic_validation_framework.py")

if __name__ == "__main__":
    main()
