"""
Test script to generate a synflow evaluation file
"""
import os
import sys

# Change to task_il directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from evo_utils import Utils
from genetic.population import Individual

# Create a test individual with sample code
class TestIndividual:
    def __init__(self):
        self.id = 'indi9999'
        self.code = [5, 32, [0, 2, 3, 5, 5], [1, 2, 5, 5, 5]]

# Test generate_synflow_file
test_indi = TestIndividual()
print("Testing generate_synflow_file()...")
print(f"Individual ID: {test_indi.id}")
print(f"Code: {test_indi.code}")

try:
    Utils.generate_synflow_file(test_indi)
    print(f"✓ Successfully generated: ./scripts/{test_indi.id}.py")

    # Read and verify the generated file
    gen_file = f'./scripts/{test_indi.id}.py'
    if os.path.exists(gen_file):
        with open(gen_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        print(f"\n--- Generated file preview (first 50 lines) ---")
        for i, line in enumerate(lines[:50], 1):
            print(f"{i:3d}: {line}")

        print(f"\n--- File stats ---")
        print(f"Total lines: {len(lines)}")

        # Check for key components
        checks = {
            "Has timestamp docstring": '"""' in content and '2026' in content,
            "Has imports": 'import torch' in content,
            "Has code variable": 'code = [5, 32,' in content,
            "Has SynflowEvaluator class": 'class SynflowEvaluator' in content,
            "Has calculate_synflow method": 'def calculate_synflow' in content,
            "Has RunModel class": 'class RunModel' in content,
            "Has do_work method": 'def do_work' in content,
        }

        print(f"\n--- Validation checks ---")
        all_passed = True
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n✓✓✓ All checks passed! File generated successfully.")
        else:
            print(f"\n✗✗✗ Some checks failed!")

    else:
        print(f"✗ Generated file not found: {gen_file}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
