import pathlib
from typing import Optional, Tuple

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


DATA_DIR = pathlib.Path(__file__).parent / "data"
PLOTS_DIR = pathlib.Path(__file__).parent / "plots"


def load_data(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
  """Load the one-factor CSV into a DataFrame.

  If no path is provided, the function will look for `data/trade_data.csv`
  inside the package.
  """
  if path is None:
    path = DATA_DIR / "trade_data.csv"
  return pd.read_csv(path)


def model_analysis(data: Optional[pd.DataFrame] = None) -> sm.regression.linear_model.RegressionResultsWrapper:
  """Run statsmodels OLS and return the fitted model.

  Args:
    data: optional DataFrame. If omitted, the function will load the default file.
  """
  if data is None:
    data = load_data()

  y, x = data["Turnover"], data["Area"]
  x = sm.add_constant(x)
  model = sm.OLS(y, x).fit()
  print("\n--- Результати регресії (statsmodels) ---")
  print(model.summary())
  return model


def model_learning(data: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
  """Train scikit-learn LinearRegression and return (intercept, coef).

  Returns:
    (b0, b1)
  """
  if data is None:
    data = load_data()

  x = data[["Area"]]
  y = data["Turnover"]
  reg = LinearRegression().fit(x, y)
  b0 = reg.intercept_
  b1 = reg.coef_[0]
  r2 = reg.score(x, y)
  print("\n--- Результати регресії (scikit-learn) ---")
  print(f"Коефіцієнт b0 (intercept): {b0:.4f}")
  print(f"Коефіцієнт b1 (Площа_X): {b1:.4f}")
  print(f"R-квадрат: {r2:.4f}")
  return b0, b1


def visualize_results(data: Optional[pd.DataFrame] = None, output_path: Optional[pathlib.Path] = None) -> pathlib.Path:
  """Create and save a regression plot. Returns the output path.

  If no output_path is provided, the plot is saved under the package `plots`
  directory as `one_factor_regression_plot.png`.
  """
  if data is None:
    data = load_data()
  b0, b1 = model_learning(data)

  PLOTS_DIR.mkdir(parents=True, exist_ok=True)
  if output_path is None:
    output_path = PLOTS_DIR / "one_factor_regression_plot.png"

  try:
    import matplotlib.pyplot as plt
    import seaborn as sns
  except Exception:  # ImportError or missing backends
    raise RuntimeError("Plotting requires matplotlib and seaborn to be installed")

  plt.figure(figsize=(10, 6))
  sns.regplot(x="Area", y="Turnover", data=data,
        scatter_kws={"color": "blue", "s": 100, "label": "Фактичні спостереження"},
        line_kws={"color": "red", "label": f"Лінія регресії: $\\hat{{Y}} = {b0:.2f} + {b1:.2f}X$"})
  plt.title("Однофакторна лінійна регресія: Товарообіг vs Торгова площа", fontsize=14)
  plt.xlabel("Торгова площа (X), тис. м$^2$", fontsize=12)
  plt.ylabel("Річний товарообіг (Y), тис. $", fontsize=12)
  plt.legend()
  plt.grid(True, linestyle="--")
  plt.savefig(output_path)
  plt.close()
  return output_path


if __name__ == "__main__":
  model_analysis()
  model_learning()
  visualize_results()
