import pathlib
from typing import Optional

import pandas as pd
import statsmodels.api as sm


DATA_DIR = pathlib.Path(__file__).parent / "data"


def load_data(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
  """Load the multifactor CSV used in the examples.

  By default this loads `data/trade_data_with_avg_visitors.csv` from the
  package directory.
  """
  if path is None:
    path = DATA_DIR / "trade_data_with_avg_visitors.csv"
  return pd.read_csv(path)


def run_regression(data: Optional[pd.DataFrame] = None) -> sm.regression.linear_model.RegressionResultsWrapper:
  """Run OLS on Turnover ~ Area + AvgVisitors and return fitted model.

  Args:
    data: optional DataFrame. If omitted, the function will load the default file.
  """
  if data is None:
    data = load_data()

  y = data["Turnover"]
  x = data[["Area", "AvgVisitors"]]
  x = sm.add_constant(x)
  model_multi = sm.OLS(y, x).fit()
  print("\n--- Результати ДВОФАКТОРНОЇ РЕГРЕСІЇ (OLS) ---")
  print(model_multi.summary())
  return model_multi


if __name__ == "__main__":
  run_regression()
