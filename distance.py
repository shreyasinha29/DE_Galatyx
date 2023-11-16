import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DistanceCalculator:
    def calculate_cosine_similarity(self, vector1, vector2):
        similarity = cosine_similarity(np.array([vector1]), np.array([vector2]))[0][0]
        return similarity
