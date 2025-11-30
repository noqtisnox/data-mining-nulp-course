import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data = np.array([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
    [1.0, 0.6], [9.0, 11.0], [0.5, 1.0], [8.5, 7.0],
    [7.0, 1.0], [6.0, 9.0], [5.5, 7.0], [4.5, 6.0]
])

initial_centers = np.array([
    [2.0, 4.0],
    [4.0, 6.0]
])

K = len(initial_centers)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

print(f"Починаємо K-Means кластеризацію на {len(data)} точках з K={K}.")
print(f"Початкові центри: {initial_centers[0]} та {initial_centers[1]}")
print("-" * 60)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f7f7f7')
ax.set_title('K-Means Кластеризація: Покрокова Візуалізація', fontsize=16, fontweight='bold')
ax.set_xlabel('Координата X', fontsize=12)
ax.set_ylabel('Координата Y', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)

scatter_data = ax.scatter(data[:, 0], data[:, 1], s=100, c='grey', alpha=0.6, edgecolors='w', linewidth=0.5, label='Точки даних')
scatter_centers = ax.scatter(initial_centers[:, 0], initial_centers[:, 1], marker='X', s=300, c=COLORS[:K], edgecolors='black', linewidth=1.5, label='Центри Кластерів')

def kmeans(data, initial_centers, max_iterations=100, tolerance=1e-4, ax=None):
    centers = initial_centers.copy()
    num_points = data.shape[0]
    K = centers.shape[0]
    labels = np.zeros(num_points, dtype=int)
    
    for iteration in range(max_iterations):
        old_centers = centers.copy()
        
        distances = np.sum((data - centers[:, np.newaxis])**2, axis=2)
        
        labels = np.argmin(distances, axis=0)

        print(f"Ітерація {iteration + 1}:")
        print(f"Мітки: {labels}")
        print("Призначення точок до центрів (Приклад 3 точок):")
        print(f"Точка {data[0]}: -> Кластер {labels[0] + 1}")
        print(f"Точка {data[2]}: -> Кластер {labels[2] + 1}")
        print(f"Точка {data[5]}: -> Кластер {labels[5] + 1}")
        
        new_centers = np.zeros_like(centers)
        counts = np.zeros(K, dtype=int)

        for i in range(num_points):
            cluster_id = labels[i]
            new_centers[cluster_id] += data[i]
            counts[cluster_id] += 1
        
        for i in range(K):
            if counts[i] > 0:
                centers[i] = new_centers[i] / counts[i]
        
        print(f"Кількість точок у кластерах: {counts}")
        for i in range(K):
            print(f"Новий Центр {i + 1}: ({centers[i, 0]:.4f}, {centers[i, 1]:.4f})")
            
        center_movement = np.sum(np.sqrt(np.sum((centers - old_centers)**2, axis=1)))
        
        print(f"Рух центрів (сумарна відстань): {center_movement:.6f}")
        
        if ax is not None:
            plot_iteration(ax, data, centers, labels, iteration + 1)
        
        print("-" * 60)
        
        if center_movement < tolerance:
            print(f"\nЗбіжність досягнута на ітерації {iteration + 1}!")
            break
            
    return centers, labels

def plot_iteration(ax, data, centers, labels, iteration):
    """Оновлює графік для поточної ітерації."""
    global scatter_data, scatter_centers, COLORS, K

    scatter_data.remove()
    scatter_centers.remove()
    
    data_colors = np.array([COLORS[label % len(COLORS)] for label in labels])
    
    scatter_data = ax.scatter(
        data[:, 0], data[:, 1],
        s=100, c=data_colors,
        alpha=0.8, edgecolors='w', linewidth=0.5,
        label='Точки даних'
    )
    
    scatter_centers = ax.scatter(
        centers[:, 0], centers[:, 1],
        marker='X', s=300, c=COLORS[:K],
        edgecolors='black', linewidth=1.5,
        label='Центри Кластерів'
    )
    
    ax.set_title(f'K-Means Кластеризація: Ітерація {iteration}', fontsize=16, fontweight='bold')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.5)
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_file = plots_dir / f"kmeans_iteration_{iteration:02d}.png"
    try:
        fig.savefig(out_file, dpi=150)
    except Exception:
        print(f"Warning: failed to save iteration plot {out_file}")

try:
    final_centers, labels = kmeans(data, initial_centers, max_iterations=100, tolerance=1e-4, ax=ax)
    
    plt.ioff()
    ax.set_title('K-Means Кластеризація: Фінальний результат', fontsize=16, fontweight='bold')
    ax.legend()
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    final_file = plots_dir / "kmeans_final.png"
    try:
        fig.savefig(final_file, dpi=200)
        print(f"Saved final plot to: {final_file}")
    except Exception:
        print("Warning: failed to save final plot")
    plt.show()

    print("\nКластеризація завершена.")
    print("Фінальні центри:")
    for i, center in enumerate(final_centers):
        print(f"Кластер {i+1}: ({center[0]:.3f}, {center[1]:.3f})")

except Exception as e:
    print(f"\nСталася помилка під час кластеризації або побудови графіку: {e}")