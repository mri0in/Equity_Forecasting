# ------------------------------------------------------------------------------
# ✅ PIPELINE WRAPPER MODULE
#
# This module exposes standardized `run_*` functions (e.g., run_training,
# run_prediction, run_optimizer) that wrap the underlying pipeline classes
# defined in the `run_*.py` modules.
#
# These wrapper functions are the **ONLY recommended entrypoints** for:
#   - The CLI (`main.py`)
#   - Automation / Production runs
#
# Why?
#   - Provides a uniform interface across all pipeline stages
#   - Ensures config-driven execution
#   - Preserves standard orchestration guarantees (logging, retries, markers)
#
# ⚠️ DO NOT bypass this module by calling the pipeline classes directly
# (e.g., ModelTrainerPipeline). Doing so may skip orchestration safeguards.
#
# If you are writing **unit tests**, you may call the underlying classes
# directly. But in **production or integration code**, always use the wrappers.
# ------------------------------------------------------------------------------
"""
src/pipeline/__init__.py

Pipeline package initializer.
        
In short:
- Tests & direct imports → use these modules and it's classes/functions directly 
-  such as(ModelTrainerPipeline for run_training.py).
- Orchestrator → use `pipeline_wrapper`
"""
# Bind submodules into the package namespace as *modules*
from . import run_training       as run_training   # module
from . import run_optimizer      as run_optimizer  # module
from . import run_walk_forward   as run_walk_forward  # module
from . import run_ensemble       as run_ensemble   # module
from . import run_prediction     as run_prediction # module


__all__ = [
    "run_training",
    "run_optimizer",
    "run_walk_forward",
    "run_ensemble",
    "run_prediction",
]
