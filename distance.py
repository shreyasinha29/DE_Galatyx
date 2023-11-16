import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DistanceCalculator:
    def calculate_cosine_similarity(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors.

        Parameters:
        - vector1, vector2: 1D arrays or lists representing vectors.

        Returns:
        - similarity: Cosine similarity between the two vectors.
        """
        similarity = cosine_similarity(np.array([vector1]), np.array([vector2]))[0][0]
        return similarity
