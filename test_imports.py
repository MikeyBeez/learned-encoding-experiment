#!/usr/bin/env python3
"""
A simple script to test if the main components can be imported without error.
"""

print("Attempting to import modules...")

try:
    from academic_validation_framework import ExperimentConfig, ExperimentRunner
    print("Successfully imported from academic_validation_framework.")

    from reporting import generate_report
    print("Successfully imported from reporting.")

    print("\n✅ All imports successful!")

except Exception as e:
    print(f"❌ An error occurred during import: {e}")
    import traceback
    traceback.print_exc()
