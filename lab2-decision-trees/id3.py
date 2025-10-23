import math
import pandas as pd
from collections import Counter

# Зчитуємо дані
df = pd.read_csv("./data/weather_data.csv")

# Функція для обчислення ентропії
def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = Counter(labels)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


# Функція для обчислення приросту інформації
def information_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    values = data[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy


# Функція для вибору найкращого атрибута
def best_attribute(data, attributes, target):
    return max(attributes, key=lambda attr: information_gain(data, attr, target))


# Функція для побудови дерева ID3
def build_tree(data, attributes, target, parent_node_class=None):
    # Якщо всі класи однакові, повертаємо вузол з цим класом
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    # Якщо немає атрибутів, повертаємо найпоширеніший клас
    if len(attributes) == 0:
        return parent_node_class or data[target].mode()[0]

    # Вибираємо найкращий атрибут
    best_attr = best_attribute(data, attributes, target)
    tree = {best_attr: {}}

    # Видаляємо вибраний атрибут з списку
    remaining_attributes = [attr for attr in attributes if attr != best_attr]

    # Рекурсивно будуємо дерево
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        subtree = build_tree(
            subset, remaining_attributes, target, data[target].mode()[0]
        )
        tree[best_attr][value] = subtree

    return tree


def run_algo():
  # Побудова дерева
  df['Температура'] = pd.to_numeric(df['Температура'])
  df['Вологість'] = pd.to_numeric(df['Вологість'])
  
  bins_temp = [0, 20, 25, 100] 
  labels_temp = ['Холодно', 'Помірно', 'Гаряче']
  df['Температура_Кат'] = pd.cut(df['Температура'], bins=bins_temp, labels=labels_temp, right=True, include_lowest=True)
  
  bins_hum = [0, 80, 200]
  labels_hum = ['Низька', 'Висока']
  df['Вологість_Кат'] = pd.cut(df['Вологість'], bins=bins_hum, labels=labels_hum, right=True, include_lowest=True)
  
  TARGET = 'Гуляти'
  attributes = ['Небо', 'Вітряно', 'Температура_Кат', 'Вологість_Кат']
  
  tree = build_tree(df, attributes, TARGET)
  print(tree)
  
  import json
  with open("./json/decision_tree_id3.json", "w") as f:
      json.dump(tree, f)


if __name__ == "__main__":
  run_algo()
