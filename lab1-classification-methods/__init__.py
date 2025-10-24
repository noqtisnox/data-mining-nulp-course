"""lab1-classification-methods package exports.

This package contains small helpers and example classification models used in the
course labs. Expose the key functions at package level for convenience.
"""

from pathlib import Path

from . import bayes as bayes

__all__ = [
	"bayes"
]

# Data directory used by the lab modules
DATA_DIR: Path = Path(__file__).parent / "data"
