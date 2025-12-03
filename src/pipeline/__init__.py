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
from . import F_training_pipeline       as F_training_pipeline   # module
from . import D_optimization_pipeline      as D_optimization_pipeline  # module
from . import G_wfv_pipeline   as G_wfv_pipeline  # module
from . import E_ensemble_pipeline       as E_ensemble_pipeline   # module
from . import H_prediction_pipeline     as H_prediction_pipeline # module


__all__ = [
    "F_training_pipeline",
    "D_optimization_pipeline",
    "G_wfv_pipeline",
    "E_ensemble_pipeline",
    "H_prediction_pipeline",
]
