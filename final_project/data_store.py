class Paper:
    def __init__(self,title,abstract,title_vector,abstract_vector):
        self.title = title
        self.abstract = abstract
        self.title_vector = title_vector
        self.abstract_vector = abstract_vector 

class Cluster:
    def __init__(self, labels, names, silhouette_score, linakge_matrix = 0):
        self.labels = labels
        self.names = names
        self.silhouette_score = silhouette_score
        self.linkage_matrix = linakge_matrix

class NammingData:
    def __init__(self, word_list, vector_list):
        self.word_list = word_list
        self.vector_list = vector_list