import re

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextBuffer:

    def __init__(self):
        self.text = ""
        self.phrases = []

    def push_text(self, text):
        """
        Adds a new chunk of text to the buffer and reanalyzes it.
        Args:
            text: The chunk of text to add.
        """
        self.text += text
        self.analyze_text()

    def consume_text(self):
        """
        Returns the latest extracted noun and adjective sequences and clears the buffer.
        Returns:
            A list of noun and adjective sequences or None if no sequences exist.
        """
        sequences = self.phrases.copy()
        self.text = ""
        self.phrases = []
        return sequences

    def analyze_text(self):
        """
        Extracts noun and adjective sequences from the buffered text.
        """
        print(f'{self.text=}')
        cleaned_text = self.clean_text(self.text)
        print(f'{cleaned_text=}')
        tagged_words = pos_tag(word_tokenize(cleaned_text))
        print(f'{tagged_words=}')
        self.phrases = self.extract_sequences(tagged_words)
        print(f'{self.phrases=}')

    def clean_text(self, text):
        """
        Cleans the text by removing special characters, lowering letters, and removing stopwords.
        Args:
            text: The text to be cleaned.
        Returns:
            The cleaned text.
        """
        # Implement your desired cleaning logic here
        lower_text = text.lower()

        # You can use regular expressions to remove special characters
        cleaned_text = re.sub(r"[^\w\s]", "", lower_text)
        stop_words = set(stopwords.words("english") + ['okay', 'uh'])
        return " ".join([word for word in cleaned_text.split() if word not in stop_words])

    def extract_sequences(self, tagged_words):
        """
        Extracts sequences of nouns and adjectives from the tagged words.
        Args:
            tagged_words: A list of (word, tag) tuples.
        Returns:
            A list of noun and adjective sequences.
        """
        noun_patterns = ["NN", "JJ", "JJR", "JJS"]
        sequences = []
        current_sequence = []
        for word, tag in tagged_words:
            if tag in noun_patterns:
                current_sequence.append(word)
            else:
                if current_sequence:
                    sequences.append(" ".join(current_sequence))
                    current_sequence = []

        # Add the last sequence if not empty
        if current_sequence:
            sequences.append(" ".join(current_sequence))
        return sequences
