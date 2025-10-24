import pandas as pd
import numpy as np
from collections import Counter
import pprint
import json


# =========================================================================
# 1. Ентропія (H(S)) - ВИКОРИСТОВУЄТЬСЯ І В ID3, І В C4.5
# =========================================================================
def calculate_entropy(data_subset: pd.DataFrame, target_attribute: str) -> float:
    """
    Обчислює ентропію для заданої підмножини даних S.
    H(S) = -Σ p(x_i) * log2(p(x_i))
    """
    class_counts = Counter(data_subset[target_attribute])
    total_samples = len(data_subset)
    entropy = 0.0

    if total_samples == 0:
        return 0.0

    for count in class_counts.values():
        p_xi = count / total_samples
        entropy -= p_xi * np.log2(p_xi)

    return entropy


# =========================================================================
# 2. Умовна Ентропія (H(S|A)) - ВИКОРИСТОВУЄТЬСЯ І В ID3, І В C4.5
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

    for value_t in data_subset[split_attribute].unique():
        subset_t = data_subset[data_subset[split_attribute] == value_t]
        p_t = len(subset_t) / total_samples
        h_t = calculate_entropy(subset_t, target_attribute)
        conditional_entropy += p_t * h_t

    return conditional_entropy


# =========================================================================
# 3. Інформаційний Приріст (Information Gain, IG(S,A)) - МЕТРИКА ID3
# =========================================================================
def calculate_information_gain(
    data_subset: pd.DataFrame, target_attribute: str, split_attribute: str
) -> float:
    """
    Обчислює інформаційний приріст IG(S,A) - **МЕТРИКА АЛГОРИТМУ ID3**.
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
# 4. Вибір найкращого атрибута для ID3 (за IG)
# =========================================================================
def get_best_split_attribute_id3(
    data_subset: pd.DataFrame, target_attribute: str, attributes: list
) -> str:
    """
    Вибирає найкращий атрибут для розщеплення, використовуючи **Information Gain (ID3)**.
    """
    best_gain = -1.0
    best_attribute = None

    for attr in attributes:
        # Обчислюємо Information Gain (IG)
        ig = calculate_information_gain(data_subset, target_attribute, attr)

        # Вибираємо атрибут з максимальним IG
        if ig > best_gain:
            best_gain = ig
            best_attribute = attr

    return best_attribute


# =========================================================================
# 5. Побудова дерева рішень ID3
# =========================================================================
def build_tree_id3(
    data_subset: pd.DataFrame, target_attribute: str, attributes: list
) -> dict:
    """
    Рекурсивно будує дерево рішень ID3.
    """

    # 1. Базові випадки (Умови зупинки)

    # a) Чиста множина (Pure set): Всі об'єкти мають однаковий клас
    if calculate_entropy(data_subset, target_attribute) == 0.0:
        # Повертаємо лист з єдиним класом
        return data_subset[target_attribute].iloc[0]

    # b) Немає більше атрибутів для розщеплення або даних
    if not attributes or len(data_subset) == 0:
        # Повертаємо лист з найпоширенішим класом
        if len(data_subset) == 0:
            # Цей випадок має бути оброблений при рекурсивному виклику, але для безпеки:
            return "No Data"
        final_class = data_subset[target_attribute].mode()[0]
        return final_class

    # 2. Вибір найкращого атрибута для розщеплення (Метрика IG)
    best_attribute = get_best_split_attribute_id3(
        data_subset, target_attribute, attributes
    )

    # Якщо найкращий атрибут не знайдено (наприклад, IG = 0 для всіх)
    if (
        not best_attribute
        or calculate_information_gain(data_subset, target_attribute, best_attribute)
        == 0.0
    ):
        final_class = data_subset[target_attribute].mode()[0]
        return final_class

    # 3. Створення вузла
    tree = {best_attribute: {}}

    # Новий список атрибутів (ID3 не допускає повторного використання атрибутів)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]

    # 4. Рекурсивне розщеплення
    for value in data_subset[best_attribute].unique():
        # Створення підмножини даних для конкретного значення атрибута
        # ID3 працює лише з дискретними атрибутами, тому відкидання колонки є стандартною практикою.
        subset = data_subset[data_subset[best_attribute] == value].drop(
            columns=[best_attribute]
        )

        # Рекурсивний виклик для підмножини
        subtree = build_tree_id3(subset, target_attribute, remaining_attributes)

        # Додавання гілки до дерева
        tree[best_attribute][value] = subtree

    return tree


def run_algo_id3():
    # Цей блок підготовки даних залишається незмінним, оскільки ID3, як і C4.5,
    # потребує дискретних (категоріальних) даних.
    try:
        df = pd.read_csv("./data/weather_data.csv")
    except FileNotFoundError:
        print(
            "\n❌ Файл './data/weather_data.csv' не знайдено. Будь ласка, переконайтеся, що він існує."
        )
        return

    # Перетворення числових даних на категоріальні (як для C4.5)
    df["Температура"] = pd.to_numeric(df["Температура"], errors="coerce")
    df["Вологість"] = pd.to_numeric(df["Вологість"], errors="coerce")
    df.dropna(inplace=True)

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
        # Для ID3 обчислюємо лише Information Gain (IG)
        ig = calculate_information_gain(df, TARGET, attr)
        results[attr] = {"IG": ig}

        print("\n--- Зведені результати ID3 ---")
        print(f"Початкова ентропія H(S) ({TARGET}): {initial_entropy:.4f}\n")

        results_df = pd.DataFrame(results).T
        results_df.columns = ["Інф. Приріст (IG)"]
        print(
            results_df.sort_values(by="Інф. Приріст (IG)", ascending=False).to_string(
                float_format="%.4f"
            )
        )

        # Вибір найкращого атрибута за IG
        best_attribute = max(results.items(), key=lambda item: item[1]["IG"])
        print(
            f"\n✅ Найкращий атрибут для першого розщеплення: **{best_attribute[0]}** з IG = **{best_attribute[1]['IG']:.4f}**"
        )

        initial_attributes = ["Небо", "Вітряно", "Температура_Кат", "Вологість_Кат"]

        print("\n--- Початок побудови дерева рішень ID3 ---\n")

        categorical_df = df.drop(columns=["Температура", "Вологість"])

        # Побудова дерева ID3
        decision_tree = build_tree_id3(categorical_df, TARGET, initial_attributes)

        # Виведення структури дерева
        print("Структура дерева рішень (JSON-подібний формат):")
        pprint.pprint(decision_tree, indent=4)

        try:
            with open("./json/decision_tree_id3.json", "w", encoding="utf-8") as f:
                json.dump(decision_tree, f, indent=4, ensure_ascii=False)
                print(
                    "\n✅ Дерево рішень успішно збережено у файл: 'decision_tree_id3.json'"
                )
        except Exception as e:
            print(f"\n❌ Помилка при збереженні дерева у JSON: {e}")


if __name__ == "__main__":
    run_algo_id3()
