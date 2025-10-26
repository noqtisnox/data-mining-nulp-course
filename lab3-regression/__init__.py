"""lab3-regression package exports.

This package contains small helpers and example regression models used in the
course labs. Expose the key functions at package level for convenience.

Notes:
- The original repo placed model modules at package root (not under a
    `models` subpackage). Import them directly to keep compatibility when the
    package is executed as a script.
"""

from pathlib import Path

# Local modules (import from package root)
from . import prepare_data as prepare_data
from . import one_factor_model as one_factor_model
from . import multifactor_model as multifactor_model

# Least-squares helpers (subpackage)
from .least_squares import least_squares_linear as least_squares_linear
from .least_squares import least_squares_nonlinear as least_squares_nonlinear
from .least_squares import compare_regressions as compare_regressions

__all__ = [
        "prepare_data",
        "one_factor_model",
        "multifactor_model",
        "least_squares_linear",
        "least_squares_nonlinear",
        "compare_regressions",
]

# Data directory used by the lab modules
DATA_DIR: Path = Path(__file__).parent / "data"