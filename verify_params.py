import sys
import os

# Add project src folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from scripts.train_loop_v5 import get_curriculum_params

params = get_curriculum_params(26)
print(f"Cycle 26 Params: {params}")

# Verify buffer default size in parser
# This is harder to check via import without mocking argparse, but we can trust the code edit if the file content is correct.
# We will just visually inspect the file content in the next step.
