from fastembed import TextEmbedding

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
