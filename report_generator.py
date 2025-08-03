#!/usr/bin/env python3
"""
Command-Line Report Generator

This script generates a markdown validation report from a JSON results file.
It serves as a command-line wrapper for the `generate_markdown_report`
function in the experiment_analysis metamodel.
"""

import argparse
import os
from experiment_analysis import generate_markdown_report

def main():
    """Main function to parse arguments and generate the report."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown validation report from a JSON results file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "json_file",
        help="Path to the input JSON results file (e.g., real_world_validation_results.json)."
    )
    parser.add_argument(
        "-o", "--output",
        default="GENERATED_VALIDATION_RESULTS.md",
        help="Path to the output markdown file."
    )

    args = parser.parse_args()

    print(f"Reading results from: {args.json_file}")

    if not os.path.exists(args.json_file):
        print(f"❌ Error: Input file not found at '{args.json_file}'")
        return

    # Call the function from the metamodel
    markdown_content = generate_markdown_report(args.json_file)

    # Save the report to the output file
    try:
        with open(args.output, 'w') as f:
            f.write(markdown_content)
        print(f"✅ Report successfully generated at: {args.output}")
    except IOError as e:
        print(f"❌ Error: Could not write to output file '{args.output}'. Reason: {e}")

if __name__ == "__main__":
    main()
