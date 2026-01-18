from flask import Flask, request, jsonify
from collections import Counter
import pickle
from google.cloud import storage
from tokenizer import *
import io
import pandas as pd
import math
import os
import numpy as np
# pip install sentence-transformers
# pip install hf_xet
from sentence_transformers import SentenceTransformer, util
from inverted_index_gcp import InvertedIndex
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# GCP Bucket configuration
BUCKET_NAME = 'ir_bucket13'
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Global variables to hold the indices and other data
body_index = None
tier1_index = None
title_index = None
anchor_index = None
id_to_title = None
pagerank = None
doc_norms = None
pageviews = None
sbert_model = None

def download_and_load_pickle(blob_name):
    """
    Function to download a pickle file from GCP Bucket and load it.
    :param blob_name:
    :return: The unpickled object.
    """
    path = f"postings_gcp/{blob_name}"

    if os.path.exists(path):
        print(f"Loading {blob_name} from local disk...")
        # print(f"DEBUG: Looking for file at: {os.path.abspath(path)}")
        # print(f"DEBUG: Current Working Directory is: {os.getcwd()}")
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {blob_name} from local disk: {e}. Re-downloading from GCP.")

    print(f"Downloading {blob_name} from GCP...")
    blob = bucket.blob(f"postings_gcp/{blob_name}")
    blob.download_to_filename(path)

    with open(path, 'rb') as f:
        return pickle.load(f)
    # Download from GCP Bucket if not found on Drive
    # print(f"Downloading {blob_name}...")
    # blob = bucket.blob(f"postings_gcp/{blob_name}")
    # contents = blob.download_as_bytes()
    # return pickle.loads(contents)

def load_pagerank():
    """
    # Load PageRank scores from GCP Bucket
    :return: dict mapping doc_id to PageRank score
    """
    print("Loading PageRank scores...")
    # Initialize an empty dictionary to hold PageRank scores
    pagerank_dict = {}
    # List all blobs with the 'pr/' prefix
    blobs = list(bucket.list_blobs(prefix='pr/'))
    # Iterate through each blob and process CSV files
    for blob in blobs:
        # Check if the blob is a CSV file
        if blob.name.endswith('.csv') or blob.name.endswith('.csv.gz'):
            # Download the blob content as bytes
            stream = blob.download_as_bytes()
            # Read the CSV content into a DataFrame
            df = pd.read_csv(io.BytesIO(stream), header=None, names=['doc_id', 'rank'], compression='gzip' if blob.name.endswith('.gz') else None)
            # Update the pagerank_dict with values from the DataFrame
            pagerank_dict.update(pd.Series(df['rank'].values, index=df.doc_id).to_dict())

    print(f"Successfully loaded PageRank for {len(pagerank_dict)} documents.")
    return pagerank_dict


def download_index_files(directory='postings_gcp'):
    """
    Download index binary files from GCP Bucket to local disk if not already present.
    :param directory:
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        print(f"Creating directory {os.path}...")
        os.makedirs(directory)

    print("Downloading index binary files to local disk...")
    # List all blobs in the specified directory
    blobs = bucket.list_blobs(prefix=f'{directory}/')
    # Iterate through each blob and download if it's a .bin file
    for blob in blobs:
        if blob.name.endswith('.bin'):
            # Determine local file path
            local_path = os.path.join(os.getcwd(), directory, os.path.basename(blob.name))
            # Skip download if file already exists
            if os.path.exists(local_path):
                continue

            print(f"Downloading {blob.name} (one-time download)...")
            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            # Download the blob to local file
            blob.download_to_filename(local_path)

    print("Finished downloading binary files.")

def initialize():
    """
    Initialize the search engine by loading indices and data from GCP Bucket.
    :return:
    """
    # Define global variables
    global body_index, tier1_index, title_index, anchor_index, id_to_title, pagerank, doc_norms, pageviews, sbert_model

    if body_index is not None:
        return

    print("Initializing Search Engine...")
    # Download index binary files to local disk
    download_index_files()

    # Load id to title mapping
    id_to_title = download_and_load_pickle('id_to_title.pkl')

    # Download and load pickled indices
    body_index = download_and_load_pickle('body_index.pkl')
    tier1_index = download_and_load_pickle('tier1_index.pkl')
    title_index = download_and_load_pickle('title_index.pkl')
    anchor_index = download_and_load_pickle('anchor_index.pkl')

    # Download and load posting locations
    body_index.posting_locs = download_and_load_pickle('body_locs.pkl')
    tier1_index.posting_locs = download_and_load_pickle('tier1_locs.pkl')
    title_index.posting_locs = download_and_load_pickle('title_locs.pkl')
    anchor_index.posting_locs = download_and_load_pickle('anchor_locs.pkl')

    # Load PageRank scores
    pagerank = load_pagerank()

    # Load document norms
    doc_norms = download_and_load_pickle('doc_norms.pkl')

    # Load pageviews
    pageviews = download_and_load_pickle('pageviews.pkl')

    sbert_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    print("SBERT loaded.\n")

    print("Initialization complete. Server is ready.")

def get_body_scores(query, n, w_body, base_dir='postings_gcp'):
    """
    Calculate body scores using TF-IDF and Cosine Similarity.
    :param query: list of query tokens
    :param n: total number of documents
    :param w_body: weight for body scores
    :param base_dir: base directory for posting lists
    :return: Counter with document scores
    """
    # Count the term frequencies in the query
    query_counts = Counter(query)
    # Define a counter to accumulate scores
    scores = Counter()

    # Iterate over each unique term in the query
    for tok, tf_q in query_counts.items():
        # Define posting list variable
        posting_list = None
        temp_list = None
        # Document frequency initialization
        df, idf_q = 0, 0
        # Check token is in the index
        if tok in tier1_index.df:
            # Retrieve the posting list for the term
            temp_list = np.array(tier1_index.read_posting_list(tok, base_dir=base_dir), dtype=np.int32)
            if len(temp_list) == 1000:
                posting_list = temp_list
                df = tier1_index.df[tok]

        # When there is no enough data in tier1
        if posting_list is None and (tok in body_index.df):
            print(f"Falling back to full body index for term: {tok} with size {len(temp_list)}") # Debug print
            # Retrieve the posting list from the full body index
            posting_list = np.array(body_index.read_posting_list(tok, base_dir=base_dir), dtype=np.int32)
            df = body_index.df[tok]


        # Update scores only if posting list is not empty
        if posting_list is not None and len(temp_list) > 0 and df > 0:

            # Separate document IDs and term frequencies
            doc_ids = posting_list[:, 0].astype(np.int32)
            tfs = posting_list[:, 1]

            # Calculate IDF using BM25 formula
            idf_q = math.log10((n - df + 0.5) / (df + 0.5) + 1)

            # Calculate term scores
            term_scores = ((tfs * (1.5 + 1)) / (tfs + 1.5) * idf_q) * w_body

            # Accumulate scores for each document
            for i in range(len(doc_ids)):
                scores[int(doc_ids[i])] += float(term_scores[i])

    return scores

def get_index_score(query_tokens, total_scores, index, N, W, base_dir):
    """
    Calculate and update scores from the title index.
    :param query_tokens:
    :param total_scores:
    :param N:
    :param W_TITLE:
    :param base_dir:
    :return:
    """
    # Iterate over each term in the query
    for term in query_tokens:
        if term in index.df:
            try:
                # Retrieve the posting list for the term from the title index
                posting_list = np.array(index.read_posting_list(term, base_dir=base_dir), dtype=np.int32)
                if len(posting_list) > 0:
                    # Extract document IDs from the posting list
                    ids = posting_list[:, 0]
                    # Calculate idf
                    tf = index.df.get(term, 1)
                    idf = math.log(N / tf)
                    # Update scores for relevant document IDs
                    for doc_id in ids:
                        if doc_id == 0:
                            continue
                        total_scores[int(doc_id)] += (W * idf)

            except KeyError:
                print(f"Term {term} not found in title index postings.")
                continue
        else:
            print(f"Term {term} not found in title index.")

    return total_scores

@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query.
        This is the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.).
        That means it is up to you to decide on whether to use stemming, remove stopwords,
        use PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []

    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # Tokenize the query with expansion
    query_tokens = tokenize(query)
    print(f"Tokenized Query: {query_tokens}")

    if not query_tokens:
        return jsonify(res)

    # Define weights for different components of the scoring
    W_TITLE = 0.2
    W_BODY = 0.3
    W_ANCHOR = 0.3
    W_PR = 0.1
    W_PV = 0.1
    W_SEMANTIC = 0.3
    N = 6348910
    base_dir = 'postings_gcp'

    # -------------------------------------------------------
    # Body Index
    # -------------------------------------------------------

    # Get scores using only the tier1 index for efficiency
    total_scores = get_body_scores(query_tokens, N, w_body=W_BODY, base_dir=base_dir)

    # -------------------------------------------------------
    # Title Index
    # -------------------------------------------------------

    total_scores = get_index_score(query_tokens, total_scores, title_index, N, W_TITLE, base_dir)

    # -------------------------------------------------------
    # Anchor Index
    # -------------------------------------------------------

    total_scores = get_index_score(query_tokens, total_scores, anchor_index, N, W_ANCHOR, base_dir)

    # -------------------------------------------------------
    # Integrate PageRank and PageView on most promising candidates
    # -------------------------------------------------------

    # Retrieve the top 1000 candidate document IDs based on accumulated scores
    top_candidate_ids = total_scores.most_common(1000)

    # Integrate PageRank and PageView into the final scoring
    final_results = []
    for doc_id, score in top_candidate_ids:
        if doc_id == 0:
            print("Final matched doc_id 0 for")
        # Retrieve PageRank and PageView scores
        pr = pagerank.get(doc_id, 0)
        pv = pageviews.get(doc_id, 0)

        # Apply logarithmic scaling to PageRank and PageView
        pr_nor = math.log10(pr * 1000 + 1)
        pv_nor = math.log10(pv + 1)

        # Combine all components to get the final score
        final_score = score + (pr_nor * W_PR) + (pv_nor * W_PV)

        # Append to final results
        final_results.append((doc_id, final_score))

    # -------------------------------------------------------
    # Sorting and Selection Top 100
    # -------------------------------------------------------

    # Sort the final results based on the combined score and select the top 100
    top_100 = sorted(final_results, key=lambda x: x[1], reverse=True)[:100]

    # -------------------------------------------------------
    # Reranking with SBERT
    # -------------------------------------------------------

    # Generate candidate titles for SBERT reranking
    candidate_doc_ids = [doc_id for doc_id, score in top_100]
    candidate_titles = [id_to_title.get(doc_id, "Unknown") for doc_id in candidate_doc_ids]

    # Calculate SBERT embeddings
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    title_embeddings = sbert_model.encode(candidate_titles, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_scores = util.cos_sim(query_embedding, title_embeddings)[0]

    # Combine SBERT scores with original scores for reranking
    reranked_results = []
    for i in range(len(candidate_doc_ids)):
        doc_id = candidate_doc_ids[i]
        original_score = top_100[i][1]
        semantic_score = cosine_scores[i].item()
        # Combine scores with weights
        final_score = ((1 - W_SEMANTIC) * original_score) + (W_SEMANTIC * semantic_score * 10)
        # Append to reranked results
        reranked_results.append((doc_id, candidate_titles[i], final_score))

    # ReSorting reranked results
    top_100 = sorted(reranked_results, key=lambda x: x[2], reverse=True)

    # Prepare the final output format (wiki_id, title)
    # res = [(str(doc_id), id_to_title.get(doc_id, "Unknow Title")) for doc_id, _ in top_100]
    res = [(str(doc_id), title) for doc_id, title, _ in top_100]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # Tokenize the query
    query_counts = Counter(tokenize_txt(query))

    candidate_scores = Counter()
    base_dir = 'postings_gcp'
    N = 6348910

    # Calculate scores for each term in the query
    for term, tf_q in query_counts.items():
        if term not in body_index.df:
          continue
        # Calculate IDF
        df = body_index.df.get(term, 1)
        idf = math.log10(N / df)
        # Calculate weight for the query term
        w_t_q = tf_q * idf
        # Retrieve posting list for the term
        try:
            posting_list = body_index.read_posting_list(term, base_dir=base_dir)
        except KeyError:
            continue
        # Update scores for each document in the posting list
        for doc_id, tf_d in posting_list:
            # Calculate weight for the document term
            w_t_d = tf_d * idf
            # Accumulate the score
            candidate_scores[doc_id] += (w_t_q * w_t_d)

    if not candidate_scores:
      return jsonify([('0', 'No Results Found'), ('0', 'No Results Found')])

    # Normalize scores by document norms
    results = []
    for doc_id, score in candidate_scores.items():
        try:
            doc_id = int(doc_id)
        except ValueError:
            raise ValueError(f"Document ID {doc_id} is not an integer.")
        # Retrieve document norm
        norm = doc_norms.get(doc_id, 1)
        final_score = score / norm
        results.append((doc_id, final_score))

    # Sort and select top 100 results
    top_100 = sorted(results, key=lambda x: x[1], reverse=True)[:100]

    # Prepare final output
    res = [(str(doc_id), id_to_title.get(doc_id, "Title not found")) for doc_id, score in top_100]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    """ - Returns ALL (not just top 100) search results that contain A QUERY WORD
          IN THE TITLE of articles, ordered in descending order of the NUMBER OF
          DISTINCT QUERY WORDS that appear in the title.
        - DO NOT use stemming.
        - DO USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
          tokenization and remove stopwords. For example, a document
          with a title that matches two distinct query words will be ranked before a
          document with a title that matches only one distinct query word,
          regardless of the number of times the term appeared in the title (or query).

        - Test this by navigating to a URL like:
          http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
          where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
          if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # Tokenize the query
    query_tokens = tokenize_txt(query)
    doc_scores = Counter()
    base_dir = 'postings_gcp'

    # Iterate over each term in the query
    for term in query_tokens:
        # Check if the term exists in the title index
        if term not in title_index.df:
            continue
        # Retrieve the posting list for the term
        try:
            posting_list = title_index.read_posting_list(term, base_dir=base_dir)
        except Exception as e:
            print(f"Error reading posting list for term {term}: {e}")
            continue
        # Update scores: +1 for each document containing the term in the title
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1
    # Sort documents by score in descending order
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert to (doc_id, title) format
    for doc_id, score in sorted_docs:
        title = id_to_title.get(doc_id, "Title Unknown")
        res.append((str(doc_id), title))

    if not res:
        return jsonify([('0', 'No Results Found'), ('0', 'No Results Found')])

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with an anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # Tokenize the query
    query_tokens = tokenize_txt(query)
    doc_scores = Counter()
    base_dir = 'postings_gcp'

    # Iterate over each term in the query
    for term in query_tokens:
        # Check if the term exists in the anchor index
        if term not in anchor_index.df:
            continue
        try:
            posting_list = anchor_index.read_posting_list(term, base_dir=base_dir)
        except Exception as e:
            print(f"Error reading posting list for term {term}: {e}")
            continue
        # Update scores for each document containing the term in the anchor text
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1
    # Sort documents by score in descending order
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert to (doc_id, title) format
    for doc_id, score in sorted_docs:
        title = id_to_title.get(doc_id, "Title Unknown")
        res.append((str(doc_id), title))

    if not res:
        return jsonify([('0', 'No Results Found'), ('0', 'No Results Found')])

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a JSON payload of the list of article ids. In python do:
          import requests          .post('http://YOUR_SERVER_DOMAIN/get_pagerank', JSON=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # Iterate over the list of IDs received from the user
    for doc_id in wiki_ids:
        # Retrieve the PageRank score from the global pagerank dictionary
        try:
            # Convert to int in case the input came as strings
            doc_id = int(doc_id)
            score = pagerank.get(doc_id, 0.0)
        except (ValueError, TypeError):
            score = 0.0

        res.append(score)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a JSON payload of the list of article ids. In python do:
          import requests          .post('http://YOUR_SERVER_DOMAIN/get_pageview', JSON=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # Iterate over the list of IDs received from the user
    for doc_id in wiki_ids:
        # Retrieve the pageview count from the global pageviews dictionary
        try:
            # Convert to int in case the input came as strings
            val = pageviews.get(int(doc_id), 0)
        except (ValueError, TypeError):
            val = 0

        res.append(val)
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # Initialize the search engine when the server starts
    initialize()
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
