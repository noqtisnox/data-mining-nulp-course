# models/one_factor_model.py
import pandas as pd
import statsmodels.api as sm
import numpy as np

def fit_linear_model(data: pd.DataFrame) -> dict:
    """
    Обчислює коефіцієнти для y = b0 + b1*x1
    """
    Y = data['Turnover']
    X = data[['Area']]
    
    # Додавання константи для b0
    X = sm.add_constant(X)
    
    # Навчання моделі
    model = sm.OLS(Y, X).fit()
    
    # Повертаємо ключові результати
    return {
        'b0': model.params['const'],
        'b1': model.params['Area'],
        'r_squared': model.rsquared,
        'summary': model.summary()
    }

def predict_linear(x_input: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """Обчислює прогнози за лінійним рівнянням."""
    return b0 + b1 * x_input