from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

"""
Reduces dimensions of embedded data.
PCA is useful as it takes the large (384+) dimension data and finds the 2 with the most variance between them.
This makes sense here because most dimensions will be very similar, as all titles are related to a single topic.

Then clusters data with different algorithms

Could also cluster on slightly more dimensions - we should text/visualize how much variance is captured per dimension of PCA
"""


class Clusterer:
    def __init__(self, data):
        self.data = np.array(data)

    def silhouette_score(self, labels):
        return silhouette_score(self.data, labels)

    def reduce_dimensions(self, n_components=2):
        print(f"Reducing dimensions to {n_components}")
        pca = PCA(n_components=n_components)
        self.data = pca.fit_transform(self.data)

    def optimal_pca_components(self, threshold=0.95, show_graph=False):
        pca = PCA()
        pca.fit(self.data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        if show_graph:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold * 100}% Threshold')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Explained Variance by Number of Components')
            plt.legend()
            plt.grid()
            plt.show()

        n_components = np.argmax(cumulative_variance >= threshold) + 1
        print(f"Optimal number of components to retain {threshold * 100:.1f}% variance: {n_components}")
        return n_components

    def test_pca(self, n_components=10):
        variance_captured = []
        for components in range(1, n_components + 1):
            print(f"Testing {components} components")
            pca = PCA(n_components=components)
            pca.fit_transform(self.data)
            variance_captured.append(sum(pca.explained_variance_ratio_))

        # line graph of variance captured by number of components
        plt.plot(range(1, n_components + 1), variance_captured)
        plt.title('Variance Explained by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Explained')
        plt.grid()
        plt.show()

    def kmeans(self, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.labels = kmeans.fit_predict(self.data)
        return self.labels

    def visualize(self, labels, algorithm):
        if self.data.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional for visualization.")

        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.title(f'2D Visualization of {algorithm} Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster')
        plt.show()

    def find_optimal_k(self, max_k=10):
        scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(self.data)
            score = silhouette_score(self.data, kmeans.labels_)
            scores.append(score)
            # print(f'k={k}, Silhouette Score={score:.4f}')

        plt.figure(figsize=(8, 6))
        plt.plot(range(2, max_k + 1), scores, marker='o')
        plt.title('Silhouette Scores for Different k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid()
        plt.show()

    # DBSCAN Implementation
    def dbscan(self, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(self.data)

    # Run and visualize DBSCAN
    def visualize_dbscan(self):
        db = DBSCAN(eps=0.3, min_samples=10).fit(self.data)
        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self.data[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = self.data[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()

    # Agglomerative Hierarchical Clustering Implementation
    def agglomerative(self, n_clusters=2):
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        return agg.fit_predict(self.data)


