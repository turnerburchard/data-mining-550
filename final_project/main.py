# https://api.crossref.org/swagger-ui/index.html#/Works/get_works

from datetime import datetime
from crossref.restful import Works, Etiquette

from cluster import Clusterer
from embed import Embedder

# temp stuff so we dont have to keep pinging api
from pickle_helpers import save_to_pkl, load_from_pkl  
import os

# TODO LIST
# TODO better topic selection - does the search work as we expect?
# TODO Get and cache many abstracts in a file
# TODO Label clusters intelligently with their meaning - unembed the centroids with KNN?
# TODO Analyze how good our clusters are (helps with below) - what metrics are used in the literature?
# TODO Determine how many clusters to use - hyperparameter tuning
# TODO Implement More clustering techniques - DBSCAN, Hierarchical Agglomerative Clustering (Maybe matches structure), SOM (good for high dimensions)
# TODO Do we just embed abstracts? Get keywords and tiles out as well?



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
    if not os.path.exists('final_project/data.pkl'):
        embedder = Embedder()
        topic = 'Software Engineering'
        query = call_api(topic, 50, 1)

        data = embedder.embed_texts([item['abstract'] for item in query])
        save_to_pkl(data)
    else:
        data = load_from_pkl()

    clusterer = Clusterer(data)
    opt_num_components = clusterer.optimal_pca_components(show_graph=True)
    clusterer.find_optimal_k()

    clusterer.reduce_dimensions(2)
    kmeans_clusters = clusterer.kmeans()
    clusterer.visualize(kmeans_clusters)

    # for item in query:
    #     print(get_info(item))
    #     print(embedder.embed_text(item['abstract']))


if __name__ == '__main__':
    main()
