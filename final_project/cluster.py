from sklearn.cluster import KMeans
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
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100}% Threshold')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Explained Variance by Number of Components')
            plt.legend()
            plt.grid()
            plt.show()

        n_components = np.argmax(cumulative_variance >= threshold) + 1
        print(f"Optimal number of components to retain {threshold*100:.1f}% variance: {n_components}")
        return n_components

    def test_pca(self, n_components=10):
        variance_captured = []
        for components in range(1, n_components+1):
            print(f"Testing {components} components")
            pca = PCA(n_components=components)
            pca.fit_transform(self.data)
            variance_captured.append(sum(pca.explained_variance_ratio_))

        # line graph of variance captured by number of components
        plt.plot(range(1, n_components+1), variance_captured)
        plt.title('Variance Explained by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Explained')
        plt.grid()
        plt.show()

    def kmeans(self, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.labels = kmeans.fit_predict(self.data)
        return self.labels

    def visualize(self, labels):
        if self.data.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional for visualization.")

        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.title('2D Visualization of Clusters')
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
            #print(f'k={k}, Silhouette Score={score:.4f}')

        plt.figure(figsize=(8, 6))
        plt.plot(range(2, max_k + 1), scores, marker='o')
        plt.title('Silhouette Scores for Different k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid()
        plt.show()