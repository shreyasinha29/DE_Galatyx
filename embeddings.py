import gensim.models as models

class WordEmbeddings:
    def __init__(self, model_path):
        self.model = self.load_embeddings(model_path)

    def load_embeddings(self, model_path):
        return models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=1000000)

    def get_embedding(self, word):
        try:
            return self.model[word]
        except KeyError:
            return None
