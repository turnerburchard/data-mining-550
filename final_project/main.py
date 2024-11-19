# https://api.crossref.org/swagger-ui/index.html#/Works/get_works

from datetime import datetime
import requests
from crossref.restful import Works, Etiquette
from fastembed import TextEmbedding
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# TODO LIST
# TODO Get and cache many abstracts
# TODO Label clusters intelligently


class Embedder:
    def __init__(self):
        self.embedding_model = TextEmbedding()
        print("Embedding model ready")

    def embed_text(self, text):
        return list(self.embedding_model.embed([text]))[0]

    def embed_texts(self, texts):
        return list(self.embedding_model.embed(texts))

class Clusterer:
    def __init__(self, data):
        self.data = np.array(data)

    def reduce_dimensions(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.data = pca.fit_transform(self.data)

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

# using library
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

def get_info(item):
    return [item['title'], item['publisher'], item['abstract']]



def main():
    embedder = Embedder()
    topic = 'Software Engineering'
    query = call_api(topic, 50, 1)

    data = embedder.embed_texts([item['abstract'] for item in query])

    clusterer = Clusterer(data)
    clusterer.reduce_dimensions(2)
    kmeans_clusters = clusterer.kmeans()
    clusterer.visualize(kmeans_clusters)

    # for item in query:
    #     print(get_info(item))
    #     print(embedder.embed_text(item['abstract']))


if __name__ == '__main__':
    main()


