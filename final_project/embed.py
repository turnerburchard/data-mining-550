from fastembed import TextEmbedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # def get_closest_word(self, vector):
    # # Load the pre-trained model

    #     # Get word embeddings and words from the model
    #     word_embeddings = self.embedding_model.get_word_vectors()
    #     words = self.embedding_model.get_words()

    #     # Compute cosine similarity between the input vector and all embeddings
    #     cos_sim = cosine_similarity([vector], word_embeddings)

    #     # Get the index of the closest word
    #     closest_idx = np.argmax(cos_sim)

    #     # Return the closest word
    #     return words[closest_idx]
