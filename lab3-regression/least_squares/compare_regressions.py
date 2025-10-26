from pathlib import Path
import importlib.util
from typing import Tuple

import numpy as np


def _load_data_from_module(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Dynamically load a module and return the `data` variable as numpy arrays.

    The module is expected to define `data` where data[0] is list of X and
    data[1] is list of Y.
    """
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    raw = getattr(module, "data")
    x = np.array(raw[0], dtype=float)
    y = np.array(raw[1], dtype=float)
    return x, y


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # returns (a, b) for y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    return a, b


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    # returns (c2, c1, c0) for y = c2*x^2 + c1*x + c0
    c2, c1, c0 = np.polyfit(x, y, deg=2)
    return c2, c1, c0


def plot_comparison(x: np.ndarray, y: np.ndarray, out_path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib is required to create the comparison plot")

    # Fit models
    a, b = fit_linear(x, y)
    c2, c1, c0 = fit_quadratic(x, y)

    # Prepare smooth x range for curves
    xs = np.linspace(x.min(), x.max(), 300)
    y_lin = a * xs + b
    y_quad = c2 * xs ** 2 + c1 * xs + c0

    # Compute SSE on the original data points (analytical comparison)
    y_lin_at_x = a * x + b
    y_quad_at_x = c2 * x ** 2 + c1 * x + c0
    sse_lin = float(np.sum((y - y_lin_at_x) ** 2))
    sse_quad = float(np.sum((y - y_quad_at_x) ** 2))

    # Plot
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="black", s=50, label="Data")
    plt.plot(xs, y_lin, color="blue", lw=2, label=f"Linear: y={a:.3f}x+{b:.3f} (SSE={sse_lin:.3f})")
    plt.plot(xs, y_quad, color="red", lw=2, linestyle="--", label=f"Quadratic: y={c2:.3f}x^2+{c1:.3f}x+{c0:.3f} (SSE={sse_quad:.3f})")
    plt.title("Linear vs Quadratic Regression")
    plt.xlabel("X (Area, тис. м^2)")
    plt.ylabel("Y (Turnover, тис. $)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    # Print SSEs for analytical comparison
    print(f"SSE Linear: {sse_lin:.6f}")
    print(f"SSE Quadratic: {sse_quad:.6f}")
    return out_path


def main():
    base = Path(__file__).parent
    linear_mod = base / "least_squares_linear.py"
    nonlinear_mod = base / "least_squares_nonlinear.py"

    # Prefer the data variable from the linear module (both modules use same data)
    x, y = _load_data_from_module(linear_mod)

    out = base / "plots" / "linear_vs_quadratic.png"
    path = plot_comparison(x, y, out)
    print(f"Saved comparison plot to: {path}")


if __name__ == "__main__":
    main()
