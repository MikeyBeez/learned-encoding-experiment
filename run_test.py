#!/usr/bin/env python3
"""
Quick test run of the enhanced learned encoding experiment.
This script serves as the main entry point for testing the new, rigorous framework.
"""

import subprocess
import sys

def main():
    """
    Run the enhanced experiment.
    Note: The new framework is pure Python and has no external dependencies.
    """
    print("🔬 Kicking off the Enhanced Learned Encoding Experiment Test Run 🔬")
    print("=" * 60)
    
    script_to_run = "run_enhanced_experiment.py"

    print(f"✅ All components are pure Python. No dependencies needed.")
    print(f"▶️  Executing `{script_to_run}`...")
    print("=" * 60)
    
    try:
        # We use subprocess to ensure it runs in a clean environment as a user would run it.
        # Using sys.executable ensures we use the same python interpreter.
        process = subprocess.run(
            [sys.executable, script_to_run],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print the output from the script, which includes the generated report.
        print(process.stdout)
        
        if process.stderr:
            print("\n--- Stderr ---")
            print(process.stderr)

        print("\n✅ Test run completed successfully!")

    except FileNotFoundError:
        print(f"❌ Error: The script `{script_to_run}` was not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: The script `{script_to_run}` failed with exit code {e.returncode}.")
        print("\n--- Stdout ---")
        print(e.stdout)
        print("\n--- Stderr ---")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
