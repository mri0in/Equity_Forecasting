"""
Test-level compatibility layer for pipeline modules.

This ensures test files can access the *original* run_* modules
instead of the wrapper functions exposed in src/pipeline/__init__.py.
"""

import importlib
import sys

# Map the wrapper name back to the original module
for mod_name in ["run_training", "run_optimizer", "run_walk_forward", "run_ensemble", "run_prediction"]:
    sys.modules[f"src.pipeline.{mod_name}"] = importlib.import_module(f"src.pipeline.{mod_name}")
