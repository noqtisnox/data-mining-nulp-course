import pandas as pd
import numpy as np
from collections import Counter


# =========================================================================
# 1. Ентропія (H(S))
# =========================================================================
def calculate_entropy(data_subset: pd.DataFrame, target_attribute: str) -> float:
    """
    Обчислює ентропію для заданої підмножини даних S.
    H(S) = -Σ p(x_i) * log2(p(x_i))
    """
    # Підрахунок частоти кожного класу в цільовому атрибуті
    class_counts = Counter(data_subset[target_attribute])
    total_samples = len(data_subset)
    entropy = 0.0

    if total_samples == 0:
        return 0.0

    # Застосування формули H(S)
    for count in class_counts.values():
        p_xi = count / total_samples
        # Уникаємо np.log2(0), хоча тут це неможливо, бо count > 0
        entropy -= p_xi * np.log2(p_xi)

    return entropy


# =========================================================================
# 2. Умовна Ентропія (H(S|A))
# =========================================================================
def calculate_conditional_entropy(
    data_subset: pd.DataFrame, target_attribute: str, split_attribute: str
) -> float:
    """
    Обчислює умовну ентропію (очікувану ентропію) H(S|A) після розщеплення за атрибутом A.
    H(S|A) = Σ p(t) * H(t)
    """
    conditional_entropy = 0.0
    total_samples = len(data_subset)

    if total_samples == 0:
        return 0.0

    # Ітерація по унікальних значеннях атрибута A (t належить T)
    for value_t in data_subset[split_attribute].unique():
        # Фільтруємо дані, щоб отримати підмножину t
        subset_t = data_subset[data_subset[split_attribute] == value_t]

        # p(t) - ймовірність (частка)
        p_t = len(subset_t) / total_samples

        # H(t) - ентропія підмножини t (рекурсивний виклик H(S))
        h_t = calculate_entropy(subset_t, target_attribute)

        conditional_entropy += p_t * h_t

    return conditional_entropy


# =========================================================================
# 3. Інформаційний Приріст (Information Gain, IG(S,A))
# =========================================================================
def calculate_information_gain(
    data_subset: pd.DataFrame, target_attribute: str, split_attribute: str
) -> float:
    """
    Обчислює інформаційний приріст IG(S,A).
    IG(S,A) = H(S) - H(S|A)
    """
    # H(S) - Ентропія до розщеплення
    initial_entropy = calculate_entropy(data_subset, target_attribute)

    # H(S|A) - Умовна ентропія після розщеплення за A
    conditional_entropy = calculate_conditional_entropy(
        data_subset, target_attribute, split_attribute
    )

    information_gain = initial_entropy - conditional_entropy

    return information_gain


# =========================================================================
# 4. Інформація Розщеплення (Split Information, SplitInfo(S,A))
#    - Ключова відмінність C4.5
# =========================================================================
def calculate_split_information(
    data_subset: pd.DataFrame, split_attribute: str
) -> float:
    """
    Обчислює інформацію розщеплення для атрибута A.
    SplitInfo(S, A) = -Σ (|S_t| / |S|) * log2(|S_t| / |S|)
    """
    total_samples = len(data_subset)
    split_info = 0.0

    if total_samples == 0:
        return 0.0

    # Ітерація по унікальних значеннях атрибута A (t належить T)
    for value_t in data_subset[split_attribute].unique():
        # |S_t| / |S| - частка (ймовірність p_t)
        p_t = len(data_subset[data_subset[split_attribute] == value_t]) / total_samples

        # - p_t * log2(p_t)
        split_info -= p_t * np.log2(p_t)

    return split_info


# =========================================================================
# 5. Приріст Відношення (Gain Ratio, GR(S,A))
#    - Метрика, яку використовує C4.5 для вибору атрибута
# =========================================================================
def calculate_gain_ratio(
    data_subset: pd.DataFrame, target_attribute: str, split_attribute: str
) -> float:
    """
    Обчислює Приріст відношення (Gain Ratio) для вибору атрибута в C4.5.
    GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
    """
    # 1. Обчислюємо Information Gain
    ig_sa = calculate_information_gain(data_subset, target_attribute, split_attribute)

    # 2. Обчислюємо Split Information
    split_info = calculate_split_information(data_subset, split_attribute)

    # 3. Gain Ratio = IG / Split Information
    # Захист від ділення на нуль: якщо SplitInfo = 0, атрибут не є корисним.
    if split_info == 0:
        return 0.0

    gain_ratio = ig_sa / split_info
    return gain_ratio


def get_best_split_attribute(
    data_subset: pd.DataFrame, target_attribute: str, attributes: list
) -> str:
    """
    Вибирає найкращий атрибут для розщеплення, використовуючи Gain Ratio.
    """
    best_gain_ratio = -1.0
    best_attribute = None

    for attr in attributes:
        # Обчислюємо Gain Ratio
        gain_ratio = calculate_gain_ratio(data_subset, target_attribute, attr)

        # Вибираємо атрибут з максимальним Gain Ratio
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_attribute = attr

    return best_attribute


def build_tree(
    data_subset: pd.DataFrame, target_attribute: str, attributes: list
) -> dict:
    """
    Рекурсивно будує дерево рішень C4.5.

    Returns:
        Словник, що представляє вузол або лист дерева.
    """

    # 1. Базові випадки (Умови зупинки)

    # a) Чиста множина (Pure set): Всі об'єкти мають однаковий клас
    if calculate_entropy(data_subset, target_attribute) == 0.0:
        # Повертаємо лист з єдиним класом
        return data_subset[target_attribute].iloc[0]

    # b) Немає більше атрибутів для розщеплення
    if not attributes or len(data_subset) == 0:
        # Повертаємо лист з найпоширенішим класом
        final_class = data_subset[target_attribute].mode()[0]
        return final_class

    # 2. Вибір найкращого атрибута для розщеплення
    best_attribute = get_best_split_attribute(data_subset, target_attribute, attributes)

    # Якщо найкращий атрибут не знайдено (наприклад, Gain Ratio = 0 для всіх)
    if not best_attribute:
        final_class = data_subset[target_attribute].mode()[0]
        return final_class

    # 3. Створення вузла
    tree = {best_attribute: {}}

    # Новий список атрибутів (виключаємо той, за яким щойно розщепили)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]

    # 4. Рекурсивне розщеплення
    for value in data_subset[best_attribute].unique():
        # Створення підмножини даних для конкретного значення атрибута
        subset = data_subset[data_subset[best_attribute] == value].drop(
            columns=[best_attribute]
        )

        # Рекурсивний виклик для підмножини
        # Зверніть увагу: на наступному рівні ми використовуємо remaining_attributes
        subtree = build_tree(subset, target_attribute, remaining_attributes)

        # Додавання гілки до дерева
        tree[best_attribute][value] = subtree

    return tree


def run_algo():
    df = pd.read_csv("./data/weather_data.csv")
    df["Температура"] = pd.to_numeric(df["Температура"])
    df["Вологість"] = pd.to_numeric(df["Вологість"])

    bins_temp = [0, 20, 25, 100]
    labels_temp = ["Холодно", "Помірно", "Гаряче"]
    df["Температура_Кат"] = pd.cut(
        df["Температура"],
        bins=bins_temp,
        labels=labels_temp,
        right=True,
        include_lowest=True,
    )

    bins_hum = [0, 80, 200]
    labels_hum = ["Низька", "Висока"]
    df["Вологість_Кат"] = pd.cut(
        df["Вологість"],
        bins=bins_hum,
        labels=labels_hum,
        right=True,
        include_lowest=True,
    )

    TARGET = "Гуляти"
    attributes = ["Небо", "Вітряно", "Температура_Кат", "Вологість_Кат"]
    results = {}

    initial_entropy = calculate_entropy(df, TARGET)

    for attr in attributes:
        gr = calculate_gain_ratio(df, TARGET, attr)
        ig = calculate_information_gain(df, TARGET, attr)
        split_info = calculate_split_information(df, attr)

        results[attr] = {"IG": ig, "SplitInfo": split_info, "GainRatio": gr}

    print("\n--- Зведені результати C4.5 ---")
    print(f"Початкова ентропія H(S) ({TARGET}): {initial_entropy:.4f}\n")

    results_df = pd.DataFrame(results).T
    results_df.columns = [
        "Інф. Приріст (IG)",
        "Інф. Розщеплення (SplitInfo)",
        "Приріст Відношення (GainRatio)",
    ]
    print(
        results_df.sort_values(
            by="Приріст Відношення (GainRatio)", ascending=False
        ).to_string(float_format="%.4f")
    )

    best_attribute = max(results.items(), key=lambda item: item[1]["GainRatio"])
    print(
        f"\n✅ Найкращий атрибут для першого розщеплення: **{best_attribute[0]}** з GR = **{best_attribute[1]['GainRatio']:.4f}**"
    )

    initial_attributes = ["Небо", "Вітряно", "Температура_Кат", "Вологість_Кат"]

    print("\n--- Початок побудови дерева рішень C4.5 ---\n")

    categorical_df = df.drop(columns=["Температура", "Вологість"])

    # Побудова дерева
    decision_tree = build_tree(categorical_df, TARGET, initial_attributes)

    # Виведення структури дерева
    print("Структура дерева рішень (JSON-подібний формат):")
    # Використовуємо pprint для гарного форматованого виведення словника
    import pprint

    pprint.pprint(decision_tree, indent=4)

    import json

    try:
        with open("./json/decision_tree_c45.json", "w", encoding="utf-8") as f:
            # Використовуємо ensure_ascii=False для коректного збереження кирилиці
            json.dump(decision_tree, f, indent=4, ensure_ascii=False)
        print("\n✅ Дерево рішень успішно збережено у файл: 'decision_tree_c45.json'")
    except Exception as e:
        print(f"\n❌ Помилка при збереженні дерева у JSON: {e}")


if __name__ == "__main__":
    run_algo()
