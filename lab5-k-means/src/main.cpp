#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <numeric>

// Define a structure for a 2D point (data and cluster center)
struct Point {
    double x;
    double y;
    int clusterId; // To store which cluster the point belongs to
};

/**
 * @brief Calculates the Euclidean distance between two points.
 * @param p1 The first point.
 * @param p2 The second point.
 * @return The Euclidean distance as a double.
 */
double euclideanDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

/**
 * @brief Performs K-Means clustering on the dataset.
 * @param data The input vector of data points.
 * @param initialCenters The initial cluster centers.
 * @param maxIterations The maximum number of iterations.
 * @param tolerance The minimum change in centers to consider convergence.
 * @return A pair containing the final cluster centers and the clustered data points.
 */
std::pair<std::vector<Point>, std::vector<Point>> runKMeans(
    std::vector<Point> data,
    const std::vector<Point>& initialCenters,
    int maxIterations = 100,
    double tolerance = 1e-4) {

    // 1. Initialization
    int K = initialCenters.size();
    if (K == 0) {
        std::cerr << "Error: No initial centers provided." << std::endl;
        return {{}, {}};
    }

    std::vector<Point> centers = initialCenters;
    std::vector<Point> oldCenters = centers;
    int iteration = 0;

    std::cout << "--- K-Means Clustering Started (K=" << K << ") ---\n";

    while (iteration < maxIterations) {
        std::cout << "\n--- Iteration " << iteration + 1 << " ---\n";

        // 2. Assignment Step: Assign each point to the nearest center
        // Clusters storage: vector of vectors of points (one vector for each cluster)
        std::vector<std::vector<Point>> clusters(K);

        for (Point& p : data) {
            double minDistance = std::numeric_limits<double>::max();
            int bestCluster = -1;

            // Find the closest center
            for (int i = 0; i < K; ++i) {
                double dist = euclideanDistance(p, centers[i]);
                if (dist < minDistance) {
                    minDistance = dist;
                    bestCluster = i;
                }
            }

            // Assign the point to the best cluster
            p.clusterId = bestCluster;
            if (bestCluster != -1) {
                clusters[bestCluster].push_back(p);
            }
        }

        // Print current assignments
        for (int i = 0; i < K; ++i) {
            std::cout << "Cluster " << i + 1 << " ("
                      << std::fixed << std::setprecision(2) << centers[i].x << ", "
                      << centers[i].y << "): "
                      << clusters[i].size() << " points\n";
        }


        // 3. Update Step: Recalculate new centers
        oldCenters = centers; // Save current centers for convergence check

        for (int i = 0; i < K; ++i) {
            if (clusters[i].empty()) {
                // Keep the center if the cluster is empty (or re-initialize if robust)
                std::cout << "Warning: Cluster " << i + 1 << " is empty. Center remains unchanged.\n";
                continue;
            }

            // Calculate the mean of all points in the cluster
            double sumX = 0.0;
            double sumY = 0.0;

            for (const auto& p : clusters[i]) {
                sumX += p.x;
                sumY += p.y;
            }

            centers[i].x = sumX / clusters[i].size();
            centers[i].y = sumY / clusters[i].size();
            centers[i].clusterId = i; // Center belongs to its own cluster ID
        }

        // 4. Convergence Check: Check if centers have moved significantly
        double totalCenterMovement = 0.0;
        for (int i = 0; i < K; ++i) {
            totalCenterMovement += euclideanDistance(centers[i], oldCenters[i]);
        }

        std::cout << "Total center movement: " << std::fixed << std::setprecision(4) << totalCenterMovement << std::endl;

        if (totalCenterMovement < tolerance) {
            std::cout << "\n*** Convergence achieved! Total movement is below tolerance ("
                      << tolerance << "). ***\n";
            break;
        }

        iteration++;
    }

    std::cout << "\n--- K-Means Clustering Finished ---\n";
    return {centers, data};
}

// Main function to run the example
int main() {
    // Set output precision
    std::cout << std::fixed << std::setprecision(2);

    // Sample Dataset (2D points)
    std::vector<Point> dataset = {
        {1.0, 2.0, -1}, {1.5, 1.8, -1}, {5.0, 8.0, -1}, {8.0, 8.0, -1},
        {1.0, 0.6, -1}, {9.0, 11.0, -1}, {0.5, 1.0, -1}, {8.5, 7.0, -1},
        {7.0, 1.0, -1}, {6.0, 9.0, -1}, {5.5, 7.0, -1}, {4.5, 6.0, -1}
    };

    // Initial Cluster Centers as specified by the user
    std::vector<Point> initialCenters = {
        {2.0, 4.0, 0}, // Center 1 (Cluster ID 0)
        {4.0, 6.0, 1}  // Center 2 (Cluster ID 1)
    };

    // Run the K-Means algorithm
    auto [finalCenters, clusteredData] = runKMeans(dataset, initialCenters);

    if (finalCenters.empty()) {
        return 1; // Error case
    }

    // --- Print Final Results ---

    std::cout << "\n=================================================\n";
    std::cout << "Final K-Means Results\n";
    std::cout << "=================================================\n";

    // Group the clustered data for easy printing
    std::vector<std::vector<Point>> finalClusters(finalCenters.size());
    for (const auto& p : clusteredData) {
        if (p.clusterId >= 0 && p.clusterId < finalCenters.size()) {
            finalClusters[p.clusterId].push_back(p);
        }
    }

    for (size_t i = 0; i < finalCenters.size(); ++i) {
        const auto& center = finalCenters[i];
        const auto& cluster = finalClusters[i];

        std::cout << "\nCluster " << i + 1 << " (Center: "
                  << std::fixed << std::setprecision(3) << center.x << ", "
                  << center.y << ")\n";
        std::cout << "-------------------------------------------------\n";
        std::cout << "Points (" << cluster.size() << " total):\n";

        for (const auto& p : cluster) {
            std::cout << "  (" << p.x << ", " << p.y << ")\n";
        }
    }

    return 0;
}
