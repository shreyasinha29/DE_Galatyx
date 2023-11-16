import pandas as pd
from gensim.models import KeyedVectors

class IOHandler:
    def read_phrases_csv(self, file_path, encodings=['utf-8', 'latin-1']):
        for encoding in encodings:
            try:
                phrases_df = pd.read_csv(file_path, encoding=encoding)
                return phrases_df
            except UnicodeDecodeError as e:
                print(f"Error decoding file '{file_path}' with encoding '{encoding}': {e}")
            except pd.errors.EmptyDataError as e:
                print(f"Error reading file '{file_path}': {e}")
        return None
        
    def write_distances_csv(self, distances, output_path):
        distances.to_csv(output_path, index=False)

    def read_word_embeddings_bin(self, model_path, limit=None):
        word_embeddings_model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=limit)
        return word_embeddings_model

    def save_word_embeddings_to_csv(self, word_embeddings_model, output_path):
        word_vectors = pd.DataFrame({word: word_embeddings_model[word] for word in word_embeddings_model.vocab})
        word_vectors.to_csv(output_path, index=False)
