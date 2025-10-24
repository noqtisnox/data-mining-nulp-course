"""lab2-decision-trees package exports.

This package contains small helpers and example decision trees models used in the
course labs. Expose the key functions at package level for convenience.
"""

from pathlib import Path

from . import id3 as id3
from . import c45 as c45
__all__ = [
	"id3",
	"c45"
]

# Data directory used by the lab modules
DATA_DIR: Path = Path(__file__).parent / "data"
