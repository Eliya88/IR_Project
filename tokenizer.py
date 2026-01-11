import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') # Open Multilingual Wordnet

# Initialize stemmer
stemmer = PorterStemmer()

# Define stopwords
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)

#
RE_WORD = re.compile(r"""[#@\w](['\-]?\w){2,24}""", re.UNICODE)

def get_synonyms(word, limit=1):
    """
    Get synonyms for a given word using WordNet.
    Limits the number of synonyms returned to a maximum of three.
    :param limit: Maximum number of synonyms to return.
    :param word: The word to find synonyms for.
    :return: A set of synonyms.
    """
    synonyms = set()
    # Iterate through synsets and lemmas to collect synonyms
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Avoid adding the original word as its own synonym
            syn_word = lemma.name().replace('_', ' ').lower()
            if syn_word != word.lower():
                synonyms.add(syn_word.lower())
            if len(synonyms) >= limit:
                return synonyms
    return synonyms

def tokenize_with_expansion(text, limit=1, expand=False):
    """
    Tokenization function that processes the input text by:
        1. Converting to lowercase
        2. Extracting words using regex
        3. Removing stopwords
        4. Optionally expanding tokens with synonyms
        5. Applying stemming to the remaining tokens
        6. Returning the list of processed tokens
    :param limit: Maximum number of synonyms to retrieve per token
    :param text: Input text to be tokenized
    :param expand: Boolean flag to indicate whether to expand tokens with synonyms
    :return: List of processed tokens
    """
    # Perform initial tokenization and stopword removal
    pure_tokens = tokenize_txt(text)

    # If no expansion is needed, return stemmed tokens directly
    if not expand:
        return [stemmer.stem(token) for token in pure_tokens]

    # Expand tokens with synonyms
    expanded_tokens = set(pure_tokens)
    for token in pure_tokens:
        # Get synonyms for the token
        syns = get_synonyms(token, limit=limit)
        # Tokenize and add each synonym to the expanded tokens set
        for syn in syns:
            words = tokenize_txt(syn)
            expanded_tokens.update(words)

    # Apply stemming to the expanded tokens and return
    return list({stemmer.stem(token) for token in expanded_tokens})

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
