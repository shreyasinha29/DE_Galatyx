from embeddings import WordEmbeddings
from process import PhraseProcessor
from distance import DistanceCalculator
from io_handler import IOHandler
import pandas as pd
from nltk import Counter
import re
import nltk
from nltk.corpus import stopwords


def main():
    #configuration
    model_path = 'data/word2vec.bin'
    phrases_path = 'data/phrases.csv'
    output_path = 'output/distances.csv'

    #initialize components
    io_handler = IOHandler()
    word_embeddings = WordEmbeddings(model_path)
    #phrase_processor = PhraseProcessor()
    distance_calculator = DistanceCalculator()

    #read and process phrases
    phrases_df = io_handler.read_phrases_csv(phrases_path)
    print("Phrases csv read")
    phrases = phrases_df['Phrases'].tolist()
    processed_phrases = phrases #placeholder for future phrase processing
    print("Pre Processed Phrases")
    
    # load word embeddings
    word_embeddings_model = io_handler.read_word_embeddings_bin(model_path, limit=1000000)
    print("Word embeddings read")

    #calculate distances
    print("L2 distance calculated")
    distances = calculate_distances(processed_phrases, word_embeddings_model, distance_calculator)

    #write distances to CSV
    print("Output written! check distances csv")
    io_handler.write_distances_csv(distances, output_path)

def calculate_distances(processed_phrases, word_embeddings_model, distance_calculator):
    distances = []
    for i, phrase1 in enumerate(processed_phrases):
        for j, phrase2 in enumerate(processed_phrases):
            if i < j:  #avoid redundant calculations
                vector1 = calculate_phrase_vector(phrase1, word_embeddings_model)
                vector2 = calculate_phrase_vector(phrase2, word_embeddings_model)
                similarity = distance_calculator.calculate_cosine_similarity(vector1, vector2)
                distances.append({'Phrase1': phrase1, 'Phrase2': phrase2, 'Similarity': similarity})
    return pd.DataFrame(distances)

def calculate_phrase_vector(phrase, word_embeddings_model):
    #calculating phase vector
    vectors = [word_embeddings_model[word] for word in phrase.split() if word in word_embeddings_model.key_to_index]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return None

if __name__ == "__main__":
    main()
