import numpy as np
import sys
from pathlib import Path

# Дані для однофакторної моделі з попереднього завдання
data = [
    # X (Торгова площа), тис. м^2
    [0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81],
    # Y (Річний товарообіг), тис. $
    [
        25.57324,
        35.18368,
        37.27008,
        38.18878,
        45.36968,
        60.48526,
        77.91288,
        89.90422,
        93.70907,
    ],
]

N = len(data[0]) # Кількість спостережень

def build_matrices(x: list[float], y: list[float]) -> tuple[list[float], list[list[float]]]:
    """
    Будує вектор вільних членів B та матрицю коефіцієнтів A
    для системи нормальних рівнянь 3x3 (для y = b0 + b1x + b2x^2).
    """
    
    # Суми для вектора B
    sum_y = sum(y)
    sum_xy = sum(a * b for a, b in zip(x, y))
    sum_x2y = sum((i**2) * j for i, j in zip(x, y))

    # Суми для матриці А
    sum_x = sum(x)
    sum_x2 = sum(i**2 for i in x)
    sum_x3 = sum(i**3 for i in x)
    sum_x4 = sum(i**4 for i in x)
    
    # Вектор B (права частина)
    B = [sum_y, sum_xy, sum_x2y]
    
    # Матриця А (коефіцієнти)
    A = [
        [N, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4]
    ]
    
    return B, A


def cramers_rule(B: list[float], A: list[list[float]]) -> list[float]:
    """
    Розв'язує систему лінійних рівнянь Ax = B за Правилом Крамера.
    Повертає коефіцієнти [b0, b1, b2].
    """
    matrix = np.array(A)
    determinant_A = np.linalg.det(matrix)
    
    if np.isclose(determinant_A, 0):
        # Використовуємо np.isclose для порівняння з нулем через особливості обчислень з плаваючою комою
        raise ValueError("Визначник системи дорівнює нулю; система не має єдиного розв'язку.")
    
    solutions = []
    # Обчислення визначників для кожної невідомої (b0, b1, b2)
    for i in range(len(B)):
        modified_matrix = matrix.copy()
        # Заміна i-го стовпця на вектор B
        modified_matrix[:, i] = B
        
        det_modified = np.linalg.det(modified_matrix)
        solutions.append(det_modified / determinant_A)
    
    return solutions


def quadratic_least_squares(x: list[float], y: list[float]):
    """
    Обчислює та виводить коефіцієнти b0, b1, b2 для y = b0 + b1x + b2x^2.
    """
    try:
        B, A = build_matrices(x, y)
        coefficients = cramers_rule(B, A)
        
        b0, b1, b2 = coefficients
        
        print("\n--- Результати Квадратичної Регресії (y = b0 + b1x + b2x^2) ---")
        print(f"Коефіцієнт b0 (Вільний член): {b0:.4f}")
        print(f"Коефіцієнт b1 (Лінійна компонента): {b1:.4f}")
        print(f"Коефіцієнт b2 (Квадратична компонента): {b2:.4f}")
        print("-" * 55)
        
        # Економічна інтерпретація
        final_equation = f"y = {b0:.4f} + {b1:.4f}x + {b2:.4f}x^2"
        print(f"Оцінене рівняння: {final_equation}")
        
        print("\nЕкономічна інтерпретація:")
        # Інтерпретація b0
        print(f"b0 ({b0:.2f}): Теоретичний товарообіг при нульовій площі (X=0).")
        # Інтерпретація b1 та b2
        if b2 > 0:
            print(f"Квадратичний термін (b2={b2:.2f}) позитивний: вплив площі (X) на товарообіг (Y) зростає зі зростанням площі, що вказує на прискорений ріст.")
        elif b2 < 0:
            print(f"Квадратичний термін (b2={b2:.2f}) негативний: вплив площі (X) на товарообіг (Y) сповільнюється зі зростанням площі (ефект насичення).")
        else:
            print("Квадратичний термін близький до нуля.")
            
    except ValueError as e:
        print(f"Помилка: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Виникла непередбачена помилка: {e}", file=sys.stderr)


def plot_quadratic_fit(x: list[float], y: list[float], coeffs: tuple[float, float, float], out_path: Path) -> Path:
    """Plot the data and the fitted quadratic curve and save to out_path.

    Args:
        x: list of x values
        y: list of y values
        coeffs: (b0, b1, b2) coefficients for y = b0 + b1*x + b2*x^2
        out_path: file path where the PNG will be saved

    Returns:
        Path to the saved image.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required to save the plot; skipping plot generation.", file=sys.stderr)
        return out_path

    b0, b1, b2 = coeffs
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    xs = np.linspace(x_arr.min(), x_arr.max(), 300)
    ys = b0 + b1 * xs + b2 * xs ** 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_arr, y_arr, color="black", s=40, label="Data")
    plt.plot(xs, ys, color="red", lw=2, label=f"Quadratic fit: y={b0:.4f}+{b1:.4f}x+{b2:.4f}x^2")
    plt.title("Quadratic Least Squares Fit")
    plt.xlabel("X (Area, тис. м^2)")
    plt.ylabel("Y (Turnover, тис. $)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved quadratic fit plot to: {out_path}")
    return out_path


def main():
    x = data[0]
    y = data[1]
    # Run quadratic least squares and then save a plot of the fit
    B, A = build_matrices(x, y)
    try:
        coefficients = cramers_rule(B, A)
        b0, b1, b2 = coefficients
        quadratic_least_squares(x, y)
        out_file = Path(__file__).parent / "plots" / "least_squares_nonlinear_plot.png"
        plot_quadratic_fit(x, y, (b0, b1, b2), out_file)
    except Exception as e:
        # Let quadratic_least_squares print the error details
        quadratic_least_squares(x, y)


if __name__ == "__main__":
    main()
