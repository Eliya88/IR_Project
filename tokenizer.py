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

# Regular expression for tokenization
RE_WORD = re.compile(r"""[#@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    """
    Tokenization function that processes the input text by:
        1. Converting to lowercase
        2. Extracting words using regex
        3. Removing stopwords
    :param text: Input text to be tokenized
    :return: List of processed tokens
    """
    # Perform initial tokenization and stopword removal
    pure_tokens = tokenize_txt(text)
    return [stemmer.stem(token) for token in pure_tokens]

def expand_query_w2v(query, model, index, top_n, threshold_df=1000):
    """
    Expand the query using a Word2Vec model by finding similar words.
    The function stems the original query tokens and adds similar words
    from the Word2Vec model if their similarity score exceeds 0.6.
    :param query: The input query string to be expanded.
    :param model: The Word2Vec model used for finding similar words.
    :param top_n: The number of top similar words to consider for each token.
    :return: A set of expanded query tokens.
    """
    # Tokenize and stem the original query tokens
    query_tokens = tokenize_txt(query)
    # Set to hold the expanded tokens including stems of original tokens
    expanded_set = set([stemmer.stem(token) for token in query_tokens])
    # Iterate through each token to find similar words
    for token in query_tokens:
        # Take the stemmed version of the token
        steam_token = stemmer.stem(token)
        # Check if the stemmed token is in the index and its document frequency is below the threshold
        if steam_token in index.df and index.df[steam_token] < threshold_df:
            # Check if the token exists in the Word2Vec model
            if model and token in model:
                # Get the top N similar words
                similar_words = model.most_similar(token, topn=top_n)
                for word, score in similar_words:
                    # Add the stemmed similar word if the similarity score is above 0.6
                    if score > 0.6:
                        # Clean the word and stem it before adding
                        clean_words = word.lower().replace('_', ' ').split()
                        for clean_word in clean_words:
                            expanded_set.add(stemmer.stem(clean_word))

    return expanded_set

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
