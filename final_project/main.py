# https://api.crossref.org/swagger-ui/index.html#/Works/get_works
import sys
from datetime import datetime
from crossref.restful import Works, Etiquette

from cluster import Clusterer
from embed import Embedder

from data_store import Paper, Cluster, NammingData
from search import find_target_cluster

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
# TODO Do we just embed abstracts? Get keywords and tiles out as well? Summarize abstracts first?


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
def get_all_papers(filename, topic):
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
                print(f"Processing batch {count} / {query.count()}")

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

        save_to_pkl(all_papers, filename)
        print(f"Saved all data to {filename}")




def main():
    # change filename here if needed
    path = 'final_project/Data/50k/'
    filename = 'data_50k.pkl'
    name_data_filename = 'name_data_50k'
    kmeans_filename = '2/kmeans50k'
    agg_filename = '2/agg50k'
    som_filename = '2/som50k'
    gauss_filename = '2/gauss50k'
    sample_size = 10000
    dimensions = 2
    rerun = False

    # get_sample(sample_size, path+filename, "Computer Science")
    # get_all_papers(filename, topic="Computer Science")

    all_papers = load_from_pkl(path+filename)

    if((not os.path.exists(path+name_data_filename)) or rerun):
        embedder = Embedder()
        title_words = list(set([word.lower() for paper in all_papers for word in paper.title.split()]))
        title_word_vectors = embedder.embed_texts(title_words)
        
        name_data = NammingData(title_words, title_word_vectors)
        save_to_pkl(name_data, path+name_data_filename)
    
    else:
        name_data = load_from_pkl(path+name_data_filename)
    data = [paper.abstract_vector for paper in all_papers]

    clusterer = Clusterer(data)
    # opt_num_components = clusterer.optimal_pca_components(show_graph=True)
    # opt_num_components = 280 #for 50k dataset
    # print(f'Optimal number of componets is {opt_num_components}')
    clusterer.reduce_dimensions(dimensions)


    # clusterer.find_optimal_k(max_k=20, interval=4)
    # return

    if ((not os.path.exists(path+kmeans_filename)) or rerun):
        kmeans_clusters = clusterer.kmeans(n_clusters=9)
        kmeans_names = clusterer.name_clusters(kmeans_clusters, name_data.word_list, name_data.vector_list)
        kmeans_silhouette_score = clusterer.silhouette_score(kmeans_clusters)
        kmeans_dbi_score = clusterer.dbi(kmeans_clusters)
        kmeans = Cluster(kmeans_clusters, kmeans_names, kmeans_silhouette_score, kmeans_dbi_score)
        save_to_pkl(kmeans, path+kmeans_filename)
    else :
        kmeans = load_from_pkl(path+kmeans_filename)
        kmeans_clusters = kmeans.labels
        kmeans_names = kmeans.names
        kmeans_silhouette_score = kmeans.silhouette_score
        kmeans_dbi_score = kmeans.dbi_score
        
    print(f"K-Means Cluster Names {kmeans_names}")
    print("K-means silhouette score: ", kmeans_silhouette_score)
    print("K-means DBI score: ", kmeans_dbi_score)
    # clusterer.visualize(kmeans_clusters, "KMeans")


    # dbscan_clusters = clusterer.dbscan()
    # clusterer.silhouette_score(dbscan_clusters)
    # runs and visualizes
    # clusterer.visualize_dbscan()

    if ((not os.path.exists(path+agg_filename)) or rerun):
        agg_clusters, linkage_matrix = clusterer.agglomerative()
        agg_names = clusterer.name_clusters(agg_clusters, name_data.word_list, name_data.vector_list)
        agg_silhouette_score = clusterer.silhouette_score(agg_clusters)
        agg_dbi_score = clusterer.dbi(agg_clusters)
        agg = Cluster(agg_clusters, agg_names, agg_silhouette_score, agg_dbi_score, linkage_matrix)
        save_to_pkl(agg, path+agg_filename)
    else :
        agg = load_from_pkl(path+agg_filename)
        agg_clusters = agg.labels
        agg_names = agg.names
        agg_silhouette_score = agg.silhouette_score
        agg_dbi_score = agg.dbi_score
        linkage_matrix = agg.linkage_matrix
        
    print(f"Agglomerative Cluster Names {agg_names}")
    print("Agglomerative silhouette score: ", agg_silhouette_score)
    print("Agglomerative DBI score: ", agg_dbi_score)
    clusterer.visualize_agglomerative(linkage_matrix, n_clusters=8)
    clusterer.visualize_dendrogram(linkage_matrix)
    
    if ((not os.path.exists(path+som_filename)) or rerun):
        som_clusters = clusterer.som(size = 2)
        som_names = clusterer.name_clusters(som_clusters, name_data.word_list, name_data.vector_list)
        som_silhouette_score = clusterer.silhouette_score(som_clusters)
        som_dbi_score = clusterer.dbi(som_clusters)
        som = Cluster(som_clusters, som_names, som_silhouette_score, som_dbi_score)
        save_to_pkl(som,path+som_filename)
    else:
        som = load_from_pkl(path+som_filename)
        som_clusters = som.labels
        som_names = som.names
        som_silhouette_score = som.silhouette_score
        som_dbi_score = som.dbi_score

    print("SOM names: ", som_names)
    print("SOM silhouette score: ", som_silhouette_score)
    print("SOM DBI score: ", som_dbi_score)
    # clusterer.visualize(som_clusters, "SOM")

    if ((not os.path.exists(path+gauss_filename)) or rerun):
        gauss_clusters = clusterer.gaussian_mixture(n_components=8)
        gauss_names = clusterer.name_clusters(gauss_clusters, name_data.word_list, name_data.vector_list)
        gauss_silhouette_score = clusterer.silhouette_score(gauss_clusters)
        gauss_dbi_score = clusterer.dbi(gauss_clusters)
        gauss = Cluster(gauss_clusters, gauss_names, gauss_silhouette_score, gauss_dbi_score)
        save_to_pkl(gauss, path+gauss_filename)
    else :
        gauss = load_from_pkl(path+gauss_filename)
        gauss_clusters = gauss.labels
        gauss_names = gauss.names
        gauss_silhouette_score = gauss.silhouette_score
        gauss_dbi_score = gauss.dbi_score
        
    print(f"Gausiian Mixture Cluster Names {gauss_names}")
    print("Gausiian Mixture silhouette score: ", gauss_silhouette_score)
    print("Gausiian Mixture DBI score: ", gauss_dbi_score)
    clusterer.visualize(gauss_clusters, "Gausiian Mixture")


    # for item in query:
    #     print(get_info(item))
    #     print(embedder.embed_text(item['abstract']))



if __name__ == '__main__':
    main()
