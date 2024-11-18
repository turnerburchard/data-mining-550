# https://api.crossref.org/swagger-ui/index.html#/Works/get_works

from datetime import datetime
import requests
from crossref.restful import Works, Etiquette
from fastembed import TextEmbedding


class Embedder:
    def __init__(self):
        self.embedding_model = TextEmbedding()
        print("Embedding model ready")

    def embed_text(self, text):
        return list(self.embedding_model.embed([text]))[0]

    def embed_texts(self, texts):
        return list(self.embedding_model.embed(texts))


# using library
def call_api(topic, sample_size=10, long_ago=1):
    etiquette = Etiquette('Research Clustering', '1.0',
                          'turnerburchard.com', 'turnerburchard@gmail.com')
    works = Works(etiquette=etiquette)

    year = datetime.now().year - long_ago

    print(f"Querying for: {topic} since {year} with {sample_size} samples")
    result = works.query(topic)

    print("Results returned: ", result.count())

    filtered = result.filter(from_pub_date=year, has_abstract='true').sample(sample_size)

    return filtered

def get_info(item):
    return [item['title'], item['publisher'], item['abstract']]



def main():
    embedder = Embedder()
    topic = 'Software Engineering'
    query = call_api(topic)
    for item in query:
        print(get_info(item))
        print(embedder.embed_text(item['abstract']))


if __name__ == '__main__':
    main()


