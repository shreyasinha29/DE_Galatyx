import gensim.models as models

class WordEmbeddings:
    def __init__(self, model_path):
        #Initialize the WordEmbeddings class with the path to the word embeddings model
        self.model = self.load_embeddings(model_path)

    def load_embeddings(self, model_path):
        '''
         Load word embeddings from the specified model_path
         Parameters:
             model_path: Path to the word embeddings model in binary format
        Returns:
            Word embeddings model
        '''
        return models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=1000000)

    def get_embedding(self, word):
        #Get the embedding vector for a given word
        # Parameters:
        #     word: The word for which to retrieve the embedding
        # Returns:
        #     The embedding vector for the word, or None if the word is not present in the model
        try:
            return self.model[word]
        except KeyError:
            # If the word is not present in the model, return None

            return None
