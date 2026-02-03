import sys
import os

# Get the directory two levels up
grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from orchestrator.seed_main import SEED
