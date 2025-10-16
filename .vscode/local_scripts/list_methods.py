# .vscode/local_scripts/list_methods.py
import sys
import importlib.util
import inspect
from pathlib import Path

# --- Arguments passed from the task ---
file_path = sys.argv[1]       # The Python file path
class_name = sys.argv[2]      # The class name to inspect

# --- Load module dynamically ---
file_path = Path(file_path).resolve()
spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# --- Get class object ---
cls = getattr(module, class_name, None)
if cls is None:
    print(f"Class '{class_name}' not found in file {file_path.name}")
    sys.exit(1)

# --- List all callable methods (exclude private and dunder methods if desired) ---
methods = [
    m for m, v in inspect.getmembers(cls, predicate=inspect.isfunction)
    if not m.startswith("_")
]

print(f"Methods in class '{class_name}':")
for method in methods:
    print(f"- {method}")
