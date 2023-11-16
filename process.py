from collections import Counter

from Levenshtein import distance


class PhraseProcessor:

    def clean_phrase(self, phrase):
         """Clean and normalize a phrase."""
        cleaned_tokens = [token.lower() for token in phrase.split()]
        cleaned_phrase = ' '.join(cleaned_tokens)
        cleaned_phrase = ''.join(char for char in cleaned_phrase if char.isalnum() or char.isspace())

        return cleaned_phrase.strip()

    def process_phrases(self, phrases):
        """Process a list of phrases."""
        processed_phrases = [self.clean_phrase(phrase) for phrase in phrases]
        processed_phrases = self.remove_duplicates(processed_phrases)
        processed_phrases = self.remove_outliers(processed_phrases)
        return processed_phrases

    def remove_duplicates(self, phrases):
        """Remove duplicate phrases."""
        return list(set(phrases))

    def remove_outliers(self, phrases):
        """Remove outlier phrases based on frequency."""
        phrase_counts = Counter(phrases)
        common_phrases = [phrase for phrase, count in phrase_counts.items() if count > 1]
        return common_phrases

    def get_closest_word(self, word, word_embeddings):
        """Find the closest word in the embeddings vocabulary using Levenshtein distance."""
        closest_word = min(word_embeddings.vocab, key=lambda x: distance(word, x))
        return closest_word
