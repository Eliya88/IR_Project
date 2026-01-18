import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download necessary NLTK resources
try:
    print("Getting stopwords...")
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()

# Define stopwords
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)

# Regular expression for tokenization
RE_WORD = re.compile(r"""[#@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    """
    Tokenization function that implements stemming on the tokens.
    :return: List of processed tokens
    """
    # Perform initial tokenization and stopword removal
    pure_tokens = tokenize_txt(text)
    return [stemmer.stem(token) for token in pure_tokens]

def tokenize_txt(text):
    """
    Tokenization function that processes the input text by:
        1. Converting to lowercase
        2. Extracting words using regex
        3. Removing stopwords
        4. Returning the list of processed tokens
    :param text: Input text to be tokenized
    :return: List of processed tokens
    """
    # Perform initial tokenization and stopword removal
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [token for token in list_of_tokens if token not in all_stopwords]
