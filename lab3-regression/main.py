"""Entry point for lab3-regression demo.

This script runs the least-squares comparison plot by delegating to the
`least_squares.compare_regressions` helper. Run it from the package folder or
the repository root.
"""

from pathlib import Path

import least_squares.compare_regressions as compare_mod


def main() -> None:
	# Ensure plots directory exists and run the comparison
	base = Path(__file__).parent / "least_squares"
	print("Running least-squares comparison...")
	compare_mod.main()

if __name__ == "__main__":
	main()
