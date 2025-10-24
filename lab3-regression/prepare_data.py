from math import sin, cos
from pathlib import Path
from typing import Union

# Constants kept to preserve original behavior
N, K = 1, 9

DATA_DIR = Path(__file__).parent / "data"


def calculate_turnover(k: int, n: int) -> float:
    """Calculate a mocked turnover value.

    Args:
        k: total number of shops (used in formula)
        n: shop index

    Returns:
        Rounded turnover value.
    """
    return round(k * ((n + 1) + sin(n)), 5)


def calculate_area(k: int, n: int) -> float:
    """Calculate a mocked shop area value."""
    return (k * n) / 100


def calculate_average_visitors(k: int, n: int) -> float:
    """Calculate a mocked average visitors value."""
    return round(((k - n) % n) * sin(k) + cos(n) * (k**2), 5)


def create_data(visitors: bool = False, target_dir: Union[str, Path] = None) -> None:
    """Create CSV data files used by the regression examples.

    The function writes files into the package `data` directory by default.

    Args:
        visitors: if True, create the file with AvgVisitors column as well.
        target_dir: optional directory to write files into (Path or str).
    """
    if target_dir is None:
        target_path = DATA_DIR
    else:
        target_path = Path(target_dir)

    target_path.mkdir(parents=True, exist_ok=True)

    if visitors:
        out_file = target_path / "trade_data_with_avg_visitors.csv"
        header = "ShopNumber,Turnover,Area,AvgVisitors\n"
    else:
        out_file = target_path / "trade_data.csv"
        header = "ShopNumber,Turnover,Area\n"

    with out_file.open("w", encoding="utf-8") as file:
        file.write(header)
        for i in range(1, K + 1):
            shop_num = str(i)
            turnover = str(calculate_turnover(K, i))
            area = str(calculate_area(K, i))
            if visitors:
                avg_visitors = calculate_average_visitors(K, i)
                avg_visitors_adjusted = str(avg_visitors) if avg_visitors >= 0 else "0.00000"
                file.write(",".join([shop_num, turnover, area, avg_visitors_adjusted]) + "\n")
            else:
                file.write(",".join([shop_num, turnover, area]) + "\n")


if __name__ == "__main__":
    create_data()
    print("Generated data with fields: [ShopNumber, Turnover, Area]")

    create_data(True)
    print("Generated data with fields: [ShopNumber, Turnover, Area, AvgVisitors]")
