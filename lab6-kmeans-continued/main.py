import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data = np.array([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
    [1.0, 0.6], [9.0, 11.0], [0.5, 1.0], [8.5, 7.0],
    [7.0, 1.0], [6.0, 9.0], [5.5, 7.0], [4.5, 6.0]
])

# We'll run K-Means for K in range 2..7. Initial centers will be chosen
# deterministically from the data for reproducibility.
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]

# Globals used/overwritten per-run by prepare_and_run
COLORS = DEFAULT_COLORS
scatter_data = None
scatter_centers = None
fig = None
ax = None
PLOTS_DIR = None
K = None


def prepare_and_run(k: int):
    """Prepare figure, initial centers and plots directory for given k, then run kmeans."""
    global COLORS, scatter_data, scatter_centers, fig, ax, PLOTS_DIR, K

    print(f"\n=== Running K-Means for K={k} ===")
    PLOTS_DIR = Path(__file__).parent / f"plots-k{k}"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # choose initial centers as random points from the dataset (without replacement)
    indices = np.random.choice(len(data), size=k, replace=False)
    initial_centers = data[indices]

    # update colors length if needed
    if len(COLORS) < k:
        # extend by cycling default colors
        COLORS = (DEFAULT_COLORS * ((k // len(DEFAULT_COLORS)) + 1))[:k]

    # create figure/axis for this run
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f7f7f7')
    ax.set_title(f'K-Means Кластеризація (K={k})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Координата X', fontsize=12)
    ax.set_ylabel('Координата Y', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)

    scatter_data = ax.scatter(data[:, 0], data[:, 1], s=100, c='grey', alpha=0.6, edgecolors='w', linewidth=0.5, label='Точки даних')
    scatter_centers = ax.scatter(initial_centers[:, 0], initial_centers[:, 1], marker='X', s=300, c=COLORS[:k], edgecolors='black', linewidth=1.5, label='Центри Кластерів')

    # set global K for plotting
    K = k

    # Save initial plot
    try:
        fig.savefig(PLOTS_DIR / f"k{k}_iteration_00.png", dpi=150)
    except Exception:
        print(f"Warning: failed to save initial plot for K={k}")

    final_centers, labels = kmeans(data, initial_centers, max_iterations=100, tolerance=1e-4, ax=ax)

    # Save final plot for this K
    try:
        fig.savefig(PLOTS_DIR / f"k{k}_final.png", dpi=200)
        print(f"Saved final plot to: {PLOTS_DIR / f'k{k}_final.png'}")
    except Exception:
        print(f"Warning: failed to save final plot for K={k}")

    plt.close(fig)
    # Compute Dunn index for analytical comparison
    try:
        dunn = compute_dunn_index(data, labels)
    except Exception:
        dunn = None

    # Save metrics
    try:
        with open(PLOTS_DIR / "metrics.txt", "w", encoding="utf-8") as mf:
            mf.write(f"K={k}\n")
            mf.write(f"initial_indices={list(indices)}\n")
            mf.write(f"final_centers={final_centers.tolist()}\n")
            mf.write(f"dunn={dunn}\n")
    except Exception:
        print(f"Warning: failed to write metrics for K={k}")

    if dunn is not None:
        print(f"Dunn index for K={k}: {dunn:.6f}")

    return final_centers, labels

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


def compute_dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Dunn index for clustering in `labels`.

    Dunn = (min inter-cluster distance) / (max intra-cluster diameter)

    Inter-cluster distance: minimum distance between points in different clusters.
    Intra-cluster diameter: maximum distance between points within the same cluster.
    """
    unique_labels = np.unique(labels)
    # precompute pairwise distances
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    # compute intra-cluster diameters
    diameters = []
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) <= 1:
            diameters.append(0.0)
            continue
        sub = dists[np.ix_(idx, idx)]
        diameters.append(np.max(sub))
    max_diameter = float(np.max(diameters)) if len(diameters) > 0 else 0.0

    # compute inter-cluster distances (minimum distance between points of different clusters)
    inter_dists = []
    for i, lab_i in enumerate(unique_labels):
        idx_i = np.where(labels == lab_i)[0]
        for lab_j in unique_labels[i+1:]:
            idx_j = np.where(labels == lab_j)[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue
            sub = dists[np.ix_(idx_i, idx_j)]
            inter_dists.append(np.min(sub))

    if len(inter_dists) == 0:
        # no inter-cluster distances (single cluster)
        return float('nan')

    min_inter = float(np.min(inter_dists))

    if max_diameter == 0:
        return float('inf')

    return min_inter / max_diameter

def run_all_K(min_k: int = 2, max_k: int = 7):
    """Run k-means for K in [min_k, max_k] and save plots per-run."""
    dunn_scores = []
    k_values = []

    for k in range(min_k, max_k + 1):
        try:
            final_centers, labels = prepare_and_run(k)
            # Collect Dunn index from metrics file
            metrics_file = Path(__file__).parent / f"plots-k{k}" / "metrics.txt"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as mf:
                        lines = mf.readlines()
                        for line in lines:
                            if line.startswith("dunn="):
                                dunn_val = float(line.split("=")[1].strip())
                                dunn_scores.append(dunn_val)
                                k_values.append(k)
                                break
                except Exception:
                    pass

            print("\nКластеризація завершена для K=", k)
            print("Фінальні центри:")
            for i, center in enumerate(final_centers):
                print(f"Кластер {i+1}: ({center[0]:.3f}, {center[1]:.3f})")
        except Exception as e:
            print(f"\nСталася помилка для K={k}: {e}")

    # Plot Dunn index vs K
    if len(k_values) > 0:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, dunn_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Dunn Index', fontsize=12)
            plt.title('Dunn Index vs Number of Clusters', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(k_values)
            
            out_file = Path(__file__).parent / "dunn_vs_k.png"
            plt.savefig(out_file, dpi=200)
            print(f"\nSaved Dunn index plot to: {out_file}")
            plt.close()
        except Exception as e:
            print(f"Warning: failed to create Dunn vs K plot: {e}")


if __name__ == "__main__":
    run_all_K(2, 7)