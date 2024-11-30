# https://api.crossref.org/swagger-ui/index.html#/Works/get_works

from datetime import datetime
from crossref.restful import Works, Etiquette

from cluster import Clusterer
from embed import Embedder

from data_store import Paper

# temp stuff so we dont have to keep pinging api
from pickle_helpers import save_to_pkl, load_from_pkl
import os

# TODO LIST
# TODO better topic selection - does the search work as we expect? does "Computer Science" return only CS papers?
# TODO Get and cache many abstracts in a file - shoot for 10,000 at least
# TODO Label clusters intelligently with their meaning - unembed the centroids with KNN?
# TODO Do we just use silhouette score? Any other metrics?
# TODO Determine how many clusters to use - hyperparameter tuning
# TODO Implement More clustering techniques - SOM (good for high dimensions) - any specific to embedded text?
# TODO Do we just embed abstracts? Get keywords and tiles out as well?


"""
Calls the crossref API to get a number of samples of papers with abstracts on given topic and timeframe
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
    filename = 'data.pkl'
    sample_size = 10000
    if not os.path.exists(filename):
    # if True:
        embedder = Embedder()
        topic = 'Computer Science'
        all_abstracts = []
        all_abstract_vectors = []
        all_titles = []
        all_title_vectors = []

        for i in range(int(sample_size/100)):
        # for i in range(1):
            print(f"Performing {i}th query")
            query = call_api(topic, 100, 1)
            
            titles = [item['title'][0] for item in query]
            abstracts = [item['abstract'] for item in query]

            all_titles.extend(titles)
            all_abstracts.extend(abstracts)

            all_title_vectors.extend(embedder.embed_texts(titles))
            all_abstract_vectors.extend(embedder.embed_texts(abstracts))

        all_papers = []
        for i in range(len(all_titles)):
            all_papers.append(Paper(all_titles[i],all_abstracts[i],all_title_vectors[i],all_abstract_vectors[i]))
        # what if we summarize each abstract with a basic LLM first?

        save_to_pkl(all_papers, filename)

    all_papers = load_from_pkl(filename)

    data = [paper.abstract_vector for paper in all_papers]

    clusterer = Clusterer(data)
    opt_num_components = clusterer.optimal_pca_components(show_graph=True)
    clusterer.find_optimal_k()

    clusterer.reduce_dimensions(2)
    kmeans_clusters = clusterer.kmeans()
    print(f"Centroids for K-Means {clusterer.cluster_centroids(kmeans_clusters)}")
    # print(f"K-Means Cluster Names {clusterer.name_clusters()}")
    clusterer.visualize(kmeans_clusters, "KMeans")
    kmeans_score = clusterer.silhouette_score(kmeans_clusters)
    print("K-means silhouette score: ", kmeans_score)

    dbscan_clusters = clusterer.dbscan()
    # clusterer.silhouette_score(dbscan_clusters)
    # runs and visualizes
    # clusterer.visualize_dbscan()


    agg_clusters = clusterer.agglomerative()
    agg_score = clusterer.silhouette_score(agg_clusters)
    print("Agglomerative silhouette score: ", agg_score)
    clusterer.visualize(agg_clusters, "Agglomerative Hierarchical Clustering")
    clusterer.visualize_dendrogram()

    som_clusters = clusterer.som(size = 2)
    som_score = clusterer.silhouette_score(som_clusters)
    print("SOM silhouette score: ", som_score)
    clusterer.visualize(som_clusters, "SOM")

    # for item in query:
    #     print(get_info(item))
    #     print(embedder.embed_text(item['abstract']))


if __name__ == '__main__':
    main()
