# conftest.py — auto-loaded by pytest before any test
# Adds the project root to sys.path so `src.*` imports resolve correctly
# regardless of which directory pytest is invoked from.
import sys
import os

# Insert project root (directory containing this file) at the front of sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
