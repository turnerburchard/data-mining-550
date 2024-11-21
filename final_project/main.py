# https://api.crossref.org/swagger-ui/index.html#/Works/get_works

from datetime import datetime
from crossref.restful import Works, Etiquette
from fastembed import TextEmbedding
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# TODO LIST
# TODO better topic selection - does the search work as we expect?
# TODO Get and cache many abstracts in a file
# TODO Label clusters intelligently with their meaning - unembed the centroids with KNN?
# TODO Analyze how good our clusters are (helps with below) - what metrics are used in the literature?
# TODO Determine how many clusters to use - hyperparameter tuning
# TODO Implement More clustering techniques - DBSCAN, Hierarchical Agglomerative Clustering (Maybe matches structure), SOM (good for high dimensions)
# TODO Do we just embed abstracts? Get keywords and tiles out as well?

"""
Calls embedding model
Currently uses fastembed, could switch to other models
"""


class Embedder:
    def __init__(self):
        self.embedding_model = TextEmbedding()
        print("Embedding model ready")

    def embed_text(self, text):
        print("Embedding text")
        return list(self.embedding_model.embed([text]))[0]

    def embed_texts(self, texts):
        print("Embedding texts")
        return list(self.embedding_model.embed(texts))


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

    def test_pca(self, n_components=10):
        variance_captured = []
        for components in range(1, n_components):
            print(f"Testing {components} components")
            pca = PCA(n_components=components)
            pca.fit_transform(self.data)
            variance_captured.append(sum(pca.explained_variance_ratio_))

        # line graph of variance captured by number of components
        plt.plot(range(1, n_components), variance_captured)
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


"""
Calls the crossref API to get a number of smaples of papers with abstracts on given topic and timeframe
"""


def call_api(topic, sample_size=100, long_ago=1):
    etiquette = Etiquette('Research Clustering', '1.0',
                          'turnerburchard.com', 'turnerburchard@gmail.com')
    works = Works(etiquette=etiquette)

    year = datetime.now().year - long_ago

    print(f"Querying for: {topic} since {year} with {sample_size} samples")
    result = works.query(topic)

    filtered = result.filter(from_pub_date=year, has_abstract='true')

    print("Results returned: ", filtered.count())

    return filtered.sample(sample_size)


"""
Gets out desired info from the crossref API response
"""


def get_info(item):
    return [item['title'], item['publisher'], item['abstract']]


def main():
    embedder = Embedder()
    topic = 'Software Engineering'
    query = call_api(topic, 50, 1)

    data = embedder.embed_texts([item['abstract'] for item in query])

    clusterer = Clusterer(data)

    clusterer.test_pca(20)

    clusterer.reduce_dimensions(2)
    kmeans_clusters = clusterer.kmeans()
    clusterer.visualize(kmeans_clusters)

    # for item in query:
    #     print(get_info(item))
    #     print(embedder.embed_text(item['abstract']))


if __name__ == '__main__':
    main()
