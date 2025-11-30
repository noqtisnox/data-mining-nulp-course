import pandas as pd
from math import pi, sqrt, cos, sin, asin

R = 6371 # Earth's radius in km

def convert_to_decimal_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """Конвертує координати DMS у десяткові градуси (DD)."""
    
    # Створення нових стовпців з десятковими градусами
    df['Latitude_DD'] = df['Latitude_Grad'] + \
                        df['Latitude_Mins'] / 60 + \
                        df['Latitude_Secs'] / 3600
    
    df['Longitude_DD'] = df['Longitude_Grad'] + \
                         df['Longitude_Mins'] / 60 + \
                         df['Longitude_Secs'] / 3600
    
    return df

def convert_to_radian(df: pd.DataFrame) -> pd.DataFrame:
    """Конвертує десяткові градуси (DD) у радіани."""
    df["Latitude_Rad"] = df["Latitude_DD"] * pi / 180
    df["Longitude_Rad"] = df["Longitude_DD"] * pi / 180
    
    print(df)
    
    return df

def haversine(phi_1: float, lambda_1: float, phi_2: float, lambda_2: float) -> float:
    """
    Застосовує формулу гаверсинуса для отримання відстані (км) між двома точками
    (координати мають бути в радіанах).
    """
    d_phi = phi_2 - phi_1
    d_lambda = lambda_2 - lambda_1
    
    # Формула гаверсинуса: a = sin²(Δφ/2) + cos(φ1)⋅cos(φ2)⋅sin²(Δλ/2)
    a = sin(d_phi / 2)**2 + cos(phi_1) * cos(phi_2) * sin(d_lambda / 2)**2
    # Формула гаверсинуса: c = 2 * asin(sqrt(a))
    c = 2 * asin(sqrt(a))
    
    return R * c


def create_distance_matrix(rad_data: pd.DataFrame) -> pd.DataFrame:
    """
    Створює матрицю попарних відстаней між містами, використовуючи функцію haversine.
    """
    cities = rad_data['Name'].tolist()
    num_cities = len(cities)
    
    # Ініціалізація матриці нулями
    distance_matrix = pd.DataFrame(0.0, index=cities, columns=cities)
    
    # Обчислення відстаней
    for i in range(num_cities):
        for j in range(i, num_cities):
            # Координати для Міста 1
            city1_lat = rad_data.iloc[i]['Latitude_Rad']
            city1_lon = rad_data.iloc[i]['Longitude_Rad']
            
            # Координати для Міста 2
            city2_lat = rad_data.iloc[j]['Latitude_Rad']
            city2_lon = rad_data.iloc[j]['Longitude_Rad']
            
            # Обчислення відстані
            distance = haversine(city1_lat, city1_lon, city2_lat, city2_lon)
            
            # Заповнення симетричної матриці
            distance_matrix.loc[cities[i], cities[j]] = distance
            distance_matrix.loc[cities[j], cities[i]] = distance
            
    return distance_matrix


def hierarchical_clustering(dist_df: pd.DataFrame):
    """
    Виконує ієрархічну кластеризацію методом "найближчого сусіда" (Single Linkage).
    Повертає матрицю зв'язків (у форматі, схожому на scipy linkage matrix).
    """
    # Створення копії матриці відстаней та списку кластерів
    current_dist_df = dist_df.copy()
    current_clusters = {i: [city] for i, city in enumerate(dist_df.columns)}
    
    # Індекси для нових кластерів починаються після початкових міст
    next_cluster_id = len(current_clusters)
    
    # Матриця зв'язків: [індекс_кластера_1, індекс_кластера_2, відстань, розмір_нового_кластера]
    linkage_matrix = []
    
    # Продовжуємо, доки не залишиться один кластер
    while len(current_dist_df) > 1:
        # 1. Знайти мінімальну відстань (найближчі сусіди)
        min_distance = float('inf')
        merge_i, merge_j = -1, -1
        
        # Перебір лише нижнього трикутника (без діагоналі)
        for i in range(len(current_dist_df.columns)):
            for j in range(i + 1, len(current_dist_df.columns)):
                if current_dist_df.iloc[i, j] < min_distance:
                    min_distance = current_dist_df.iloc[i, j]
                    merge_i, merge_j = i, j

        # Отримання імен кластерів, які об'єднуються
        cluster_name_1 = current_dist_df.columns[merge_i]
        cluster_name_2 = current_dist_df.columns[merge_j]

        # 2. Оновлення матриці зв'язків (Linkage Matrix)
        # Знаходимо початкові ID (0 до N-1) або об'єднані ID (N і далі)
        # Знаходимо кластери, що містять точні назви міст/кластерів
        id1 = None
        id2 = None
        for k, v in current_clusters.items():
            if cluster_name_1 in v:
                id1 = k
            if cluster_name_2 in v:
                id2 = k
            if id1 is not None and id2 is not None:
                break

        if id1 is None or id2 is None:
            raise ValueError(f"Could not find clusters for {cluster_name_1} or {cluster_name_2}")
        
        # Створення запису про об'єднання
        new_cluster_size = len(current_clusters[id1]) + len(current_clusters[id2])
        linkage_matrix.append([min(id1, id2), max(id1, id2), min_distance, new_cluster_size])
        
        # 3. Об'єднання кластерів
        new_cluster_name = f"Cluster_{next_cluster_id}"
        
        # Оновлення списку кластерів
        current_clusters[next_cluster_id] = current_clusters.pop(id1) + current_clusters.pop(id2)
        
        # 4. Оновлення матриці відстаней (метод Single Linkage)
        new_row_distances = {}
        cols_to_keep = [col for col in current_dist_df.columns if col not in [cluster_name_1, cluster_name_2]]
        
        # Обчислення відстані нового кластера до решти (мін. відстань)
        for other_cluster in cols_to_keep:
            # Single Linkage: min(D(A, C), D(B, C))
            dist_to_other = min(current_dist_df.loc[cluster_name_1, other_cluster],
                                current_dist_df.loc[cluster_name_2, other_cluster])
            new_row_distances[other_cluster] = dist_to_other

        # Створення нової матриці
        new_columns = [new_cluster_name] + cols_to_keep
        new_dist_df = pd.DataFrame(0.0, index=new_columns, columns=new_columns)
        
        # Копіювання існуючих відстаней
        new_dist_df.loc[cols_to_keep, cols_to_keep] = current_dist_df.loc[cols_to_keep, cols_to_keep]
        
        # Заповнення нових відстаней
        for col, dist in new_row_distances.items():
            new_dist_df.loc[new_cluster_name, col] = dist
            new_dist_df.loc[col, new_cluster_name] = dist
        
        current_dist_df = new_dist_df
        next_cluster_id += 1
        
    # Повертаємо матрицю зв'язків як DataFrame для зручності
    return pd.DataFrame(linkage_matrix, columns=['cluster1_idx', 'cluster2_idx', 'distance', 'new_cluster_size'])


if __name__ == "__main__":
    data = pd.read_csv("./data/cities.csv")
    
    # Послідовне перетворення даних
    dd_data = convert_to_decimal_degrees(data.copy())
    rad_data = convert_to_radian(dd_data)
    
    # Створення матриці відстаней
    distance_df = create_distance_matrix(rad_data)
    
    print("Матриця попарних відстаней (км):")
    print(distance_df.round(2))

    # Запуск ієрархічної кластеризації
    linkage_result_df = hierarchical_clustering(distance_df)
    
    print("\nМатриця зв'язків (Linkage Matrix) для кластеризації:")
    print(linkage_result_df.round(2))
