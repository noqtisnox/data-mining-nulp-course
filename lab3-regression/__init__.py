"""lab3-regression package exports.

This package contains small helpers and example regression models used in the
course labs. Expose the key functions at package level for convenience.
"""

from pathlib import Path

from . import prepare_data as prepare_data
from . import one_factor_model as one_factor_model
from . import multifactor_model as multifactor_model

__all__ = [
	"prepare_data",
	"one_factor_model",
	"multifactor_model",
]

# Data directory used by the lab modules
DATA_DIR: Path = Path(__file__).parent / "data"
