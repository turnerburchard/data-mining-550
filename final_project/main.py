import sys
from datetime import datetime
from crossref.restful import Works, Etiquette
from cluster import Clusterer
from embed import Embedder
from data_store import Paper
from search import find_target_cluster
from pickle_helpers import save_to_pkl, load_from_pkl
import os
import random

"""
Calls the crossref API to get a number of samples of papers with abstracts on given topic and timeframe
"""
def call_api(topic, sample_size=100, years_ago=1):
    etiquette = Etiquette('Research Clustering', '1.0',
                          'turnerburchard.com', 'turnerburchard@gmail.com')
    works = Works(etiquette=etiquette)

    year = datetime.now().year - years_ago

    print(f"Querying for: {topic} since {year} with {sample_size} samples")
    result = works.query(topic)

    filtered = result.filter(from_pub_date=year, has_abstract='true')

    print("Results returned: ", filtered.count())

    if sample_size > 0:
        return filtered.sample(sample_size)
    elif sample_size == 0:
        return filtered


"""
Gets out desired info from the crossref API response
"""


def get_info(item):
    return [item['title'], item['publisher'], item['abstract']]


# get a sample of the papers returned for a given topic
# for testing only
def get_sample(sample_size, filename, topic):
    if not os.path.exists(filename):
        # if True:
        embedder = Embedder()
        all_abstracts = []
        all_abstract_vectors = []
        all_titles = []
        all_title_vectors = []

        for i in range(int(sample_size / 100)):
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
            all_papers.append(Paper(all_titles[i], all_abstracts[i], all_title_vectors[i], all_abstract_vectors[i]))

        save_to_pkl(all_papers, filename)


# Get all papers that are returned by api for a given topic
def get_all_papers(sample_size, filename, topic):
    if not os.path.exists(filename):
        embedder = Embedder()
        all_papers = []

        query = call_api(topic, 0, 1)  # Single call to initialize the query object
        batch_size = 100
        batch_abstracts = []
        batch_titles = []
        count = 0

        # Process the query object in batches
        for item in query:
            count += 1
            batch_abstracts.append(item['abstract'])
            batch_titles.append(item['title'][0])

            # When a batch is full, embed and clear the batch
            if len(batch_abstracts) == batch_size:
                print(f"Processing batch {count} / {sample_size}")

                # Embed the current batch
                batch_abstract_vectors = embedder.embed_texts(batch_abstracts)
                batch_title_vectors = embedder.embed_texts(batch_titles)

                # Append embeddings and raw text to cumulative lists
                for i in range(len(batch_abstracts)):
                    all_papers.append(Paper(
                        batch_titles[i],
                        batch_abstracts[i],
                        batch_title_vectors[i],
                        batch_abstract_vectors[i]
                    ))

                # Clear the batch lists
                batch_abstracts = []
                batch_titles = []

            if count > sample_size:
                break

        save_to_pkl(all_papers, filename)
        print(f"Saved all data to {filename}")




def main():
    # change filename here if needed
    filename = 'esofdata.pkl'
    sample_size = 10000

    # get_sample(sample_size, filename, "Computer Science")
    # get_all_papers(sample_size, filename, topic="Software")

    all_papers = load_from_pkl(filename)

    embedder = Embedder()

    title_words = [paper.title for paper in all_papers]
    title_word_vectors = [paper.title_vector for paper in all_papers]

    all_data = [paper.abstract_vector for paper in all_papers]

    data = random.sample(all_data, 500)


    clusterer = Clusterer(data)
    opt_num_components = clusterer.optimal_pca_components(show_graph=True)
    clusterer.find_optimal_k()

    clusterer.reduce_dimensions(2)
    kmeans_clusters = clusterer.kmeans(7)
    # print(f"Centroids for K-Means {clusterer.cluster_centroids(kmeans_clusters)}")
    print(f"K-Means Cluster Names {clusterer.name_clusters(kmeans_clusters,title_words, title_word_vectors)}")
    cluster_names = ["Case Study", "SLR", "Meta-Analysis", "Empirical Study", "Experiment", "Simulation",
                     "Tool Evaluation"]
    clusterer.visualize(kmeans_clusters, "KMeans", cluster_names)
    kmeans_score = clusterer.silhouette_score(kmeans_clusters)
    print("K-means silhouette score: ", kmeans_score)

    # dbscan_clusters = clusterer.dbscan()
    # unique_dbscan = set(dbscan_clusters)
    #
    # # calculate silhouette score for dbscan:
    #
    # dbscan_data = clusterer.data[dbscan_clusters != -1]
    # dbscan_labels = dbscan_clusters[dbscan_clusters != -1]
    #
    # if len(set(dbscan_labels)) > 1:  # At least two clusters required
    #     silhouette = clusterer.silhouette_score(dbscan_labels)
    #     print("Silhouette Score:", silhouette)
    # else:
    #     print("Silhouette Score could not be calculated (insufficient clusters).")
    #
    #
    # # runs and visualizes
    # # clusterer.visualize_dbscan()
    minibatch_kmeans = clusterer.minibatch_kmeans()
    minibatch_score = clusterer.silhouette_score(minibatch_kmeans)
    print("Minibatch K-means silhouette score: ", minibatch_score)
    #
    #
    agg_clusters, linkage_matrix = clusterer.agglomerative()
    agg_score = clusterer.silhouette_score(agg_clusters)
    print("Agglomerative silhouette score: ", agg_score)
    # clusterer.visualize(agg_clusters, "Agglomerative Hierarchical Clustering")
    # clusterer.visualize_dendrogram(linkage_matrix)
    #
    som_clusters = clusterer.som(size = 2)
    som_score = clusterer.silhouette_score(som_clusters)
    print("SOM silhouette score: ", som_score)
    # clusterer.visualize(som_clusters, "SOM")
    #
    birch_clusters = clusterer.birch()
    #birch_score = clusterer.silhouette_score(birch_clusters)
    #3print("Birch silhouette score: ", birch_score)
    # clusterer.visualize(birch_clusters, "Birch")
    #
    gaussian_clusters = clusterer.gaussian_mixture()
    gaussian_score = clusterer.silhouette_score(gaussian_clusters)
    print("Gaussian silhouette score: ", gaussian_score)
    # clusterer.visualize(gaussian_clusters, "Gaussian Mixture")


if __name__ == '__main__':
    main()
    # pass

