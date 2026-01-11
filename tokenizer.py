import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download
nltk.download('stopwords')

#
stemmer = PorterStemmer()
#
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)

#
RE_WORD = re.compile(r"""[#@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize_txt(text):
    """

    """
    #
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [stemmer.stem(token) for token in list_of_tokens if token not in all_stopwords]
