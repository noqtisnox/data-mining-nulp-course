data = [
    [0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81],
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

N = 9


def find_a(x, y):
    x_squared = [i**2 for i in x]
    xy = [a * b for a, b in zip(x, y)]

    numerator = N * sum(xy) - sum(x) * sum(y)
    denominator = N * sum(x_squared) - (sum(x) ** 2)

    return numerator / denominator


def find_b(a, x, y):
    return (sum(y) - a * sum(x)) / N


def least_squares_method(x, y):
    a = find_a(x, y)
    b = find_b(a, x, y)

    print("Using 'Least Squares Method'")
    print(f"For each sq.m. (X) total turnover (Y) increases by ~{a:.2f}k$")
    print(f"Theoretically, without any area (X = 0) total turnover (Y) would be ~{b:.2f}k$")
    print(f"Final equation: y={a:.2f}x+{b:.2f}")


def main():
    x = data[0]
    y = data[1]
    least_squares_method(x, y)


if __name__ == "__main__":
    main()
