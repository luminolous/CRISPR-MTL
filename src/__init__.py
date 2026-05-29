# Make `from src import dataset, model, train, evaluate` work from the project root.
# Adds src/ to sys.path so intra-package imports (e.g. `from model import ...`)
# resolve correctly whether code is run from src/ or the project root.
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
