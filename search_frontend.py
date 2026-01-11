from flask import Flask, request, jsonify
from collections import Counter
import pickle
from google.cloud import storage
from tokenizer import *
import io
import pandas as pd
import math
import os
import time
import numpy as np
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


def download_and_load_pickle(blob_name):
    """
    Function to download a pickle file from GCP Bucket and load it.
    :param blob_name:
    :return: The unpickled object.
    """
    # Download from GCP Bucket if not found on Drive
    print(f"Downloading {blob_name}...")
    blob = bucket.blob(f"postings_gcp/{blob_name}")
    contents = blob.download_as_bytes()
    return pickle.loads(contents)

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
            #local_path = os.path.join('/content', blob.name)
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

def normalize_index_paths(index):
    """
    מסירה קידומות נתיב מיותרות מרשימות המיקומים של האינדקס
    כדי להבטיח עבודה אחידה מול תיקיית postings_gcp
    """
    for term in index.posting_locs:
        # כל מיקום הוא טאפל של (path, offset)
        # os.path.basename משאיר רק את שם הקובץ (למשל 'file.bin')
        index.posting_locs[term] = [(os.path.basename(loc[0]), loc[1]) for loc in index.posting_locs[term]]

def initialize():
    """
    Initialize the search engine by loading indices and data from GCP Bucket.
    :return:
    """
    # Define global variables
    global body_index, tier1_index, title_index, anchor_index, id_to_title, pagerank, doc_norms, pageviews

    print("Initializing Search Engine...")
    # Download index binary files to local disk
    download_index_files()

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

    # Normalize index paths
    normalize_index_paths(body_index)
    normalize_index_paths(tier1_index)
    normalize_index_paths(title_index)
    normalize_index_paths(anchor_index)

    # Load id to title mapping
    id_to_title = download_and_load_pickle('id_to_title.pkl')

    # Load PageRank scores
    pagerank = load_pagerank()

    # Load document norms
    doc_norms = download_and_load_pickle('doc_norms.pkl')

    # Load pageviews
    pageviews = download_and_load_pickle('pageviews.pkl')

    print("Initialization complete. Server is ready.")

# Initialize the search engine when the server starts
initialize()




def get_body_scores(query, n, w_body):
    """
    Calculate body scores using TF-IDF and Cosine Similarity.
    :param query: list of query tokens
    :param n: total number of documents
    :param w_body: weight for body scores
    :return: Counter with document scores
    """
    # Count the term frequencies in the query
    query_counts = Counter(query)
    # Define a counter to accumulate scores
    scores = Counter()
    # Define base directory for postings
    base_dir = 'postings_gcp'

    # Iterate over each unique term in the query
    for tok, tf_q in query_counts.items():
        # Define posting list variable
        posting_list = None
        # Document frequency initialization
        df = 0
        # Check token is in the index
        if tok in tier1_index.df:
            # Retrieve the posting list for the term
            temp_list = np.array(tier1_index.read_posting_list(tok, base_dir=base_dir), dtype=np.int32)
            print(f"Tier1 index found for term: {tok} - list size: {len(temp_list)}")
            if len(temp_list) >= 100:
                posting_list = temp_list
                df = tier1_index.df[tok]

        # When there is no enough data in tier1
        if posting_list is None and (tok in body_index.df):
            print(f"Falling back to full body index for term: {tok} with size {len(temp_list)}")
            # Retrieve the posting list from the full body index
            posting_list = np.array(body_index.read_posting_list(tok, base_dir=base_dir), dtype=np.int32)
            df = body_index.df[tok]

        # Update scores only if posting list is not empty
        if posting_list is not None and len(temp_list) > 0 and df > 0:

            # Separate document IDs and term frequencies
            doc_ids = posting_list[:, 0].astype(np.int32)
            tfs = posting_list[:, 1]

            # Calculate IDF and weight for the query term
            idf = np.log10(n / df)
            w_t_q = tf_q * idf

            norms = np.array([doc_norms.get(int(d), 1.0) for d in doc_ids], dtype=np.float32)

            # Calculate term scores
            term_scores = ((tfs * idf) * w_t_q / norms) * w_body

            # Accumulate scores for each document
            for i in range(len(doc_ids)):
                scores[int(doc_ids[i])] += float(term_scores[i])

    return scores

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
    # BEGIN SOLUTION

    # Tokenize the query using the provided tokenizer
    query_tokens = tokenize_txt(query)
    if not query_tokens:
        return jsonify(res)

    # Define weights for different components of the scoring
    W_TITLE = 0.5
    W_BODY = 0.2
    W_ANCHOR = 0.3
    W_PR = 0.1
    W_PV = 0.1
    N = 6348910
    base_dir = 'postings_gcp'

    # -------------------------------------------------------
    # Body Index
    # -------------------------------------------------------
    t_start = time.time()  # -------------------------

    # Get scores using only the tier1 index for efficiency
    total_scores = get_body_scores(query_tokens, N, w_body=W_BODY)

    # Create a set of candidate document IDs from body scores
    candidate_ids = np.array(list(total_scores.keys()), dtype=np.int32)

    t_body = time.time()  # -------------------------
    print(f"[PERF] Body scoring took: {t_body - t_start:.4f}s Candidates: {len(total_scores)}")
    # -------------------------------------------------------
    # Title Index
    # -------------------------------------------------------
    t_start = time.time()  # -------------------------

    # Iterate over each term in the query
    for term in query_tokens:
        if term in title_index.df:
            try:
                # Retrieve the posting list for the term from the title index
                posting_list = np.array(title_index.read_posting_list(term, base_dir=base_dir), dtype=np.int32)
                print(f"Title index found for term: {term} - list size: {len(posting_list)}")
                if len(posting_list) > 0:
                    # Extract document IDs from the posting list
                    title_ids = posting_list[:, 0]
                    # Find intersection with candidate IDs
                    relevant_ids = np.intersect1d(title_ids, candidate_ids, assume_unique=True)
                    # Update scores for relevant document IDs
                    for doc_id in relevant_ids:
                        total_scores[int(doc_id)] += W_TITLE

            except KeyError:
                print(f"Term {term} not found in title index postings.")
                continue
        else:
            print(f"Term {term} not found in title index.")

    extra_end = time.time()
    print(f"[PERF] Title took: {extra_end - t_start:.4f}s")

    # -------------------------------------------------------
    # Anchor Index
    # -------------------------------------------------------
    t_start = time.time()  # -------------------------

    # Iterate over each term in the query
    for term in query_tokens:
        if term in anchor_index.df:
            try:
                posting_list = np.array(anchor_index.read_posting_list(term, base_dir=base_dir), dtype=np.int32)
                print(f"Anchor index found for term: {term} - list size: {len(posting_list)}")

                if len(posting_list) > 0:
                    # Extract document IDs from the posting list
                    doc_ids = posting_list[:, 0]
                    # Find intersection with candidate IDs
                    relevant_ids = np.intersect1d(doc_ids, candidate_ids, assume_unique=True)
                    # Update scores for relevant document IDs
                    for doc_id in relevant_ids:
                        total_scores[int(doc_id)] += W_ANCHOR

            except KeyError:
                print(f"Term {term} not found in anchor index postings.")
                continue
        else:
            print(f"Term {term} not found in anchor index.")

    extra_end = time.time()
    print(f"[PERF] Anchor scoring took: {extra_end - t_start:.4f}s")

    # Retrieve the top 1000 candidate document IDs based on accumulated scores
    top_candidate_ids = total_scores.most_common(1000)

    # Integrate PageRank and PageView into the final scoring
    final_results = []
    for doc_id, score in top_candidate_ids:
        # Retrieve PageRank and PageView scores
        pr = pagerank.get(doc_id, 0.00000001)
        pv = pageviews.get(doc_id, 0)

        # Apply logarithmic scaling to PageRank and PageView
        pr_nor = max(0.0, math.log10(pr) + 8)
        pv_nor = math.log10(pv + 1) / 7.0
        # Combine all components to get the final score
        final_score = score + (pr_nor * W_PR) + (pv_nor * W_PV)
        # Append to final results
        final_results.append((doc_id, final_score))

    # -------------------------------------------------------
    # Final Sorting and Selection
    # -------------------------------------------------------

    # Sort the final results based on the combined score and select the top 100
    top_100 = sorted(final_results, key=lambda x: x[1], reverse=True)[:100]

    # Prepare the final output format (wiki_id, title)
    print(top_100[0])
    res = [(str(doc_id), id_to_title.get(doc_id, "Unknown Title")) for doc_id, score in top_100]

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

    # ספירת תדירויות בשאילתה (tf_q)
    query_counts = Counter(tokenize_txt(query))

    # מילון צובר לציונים: {doc_id: accumulated_score}
    candidate_scores = Counter()

    # N - מספר המסמכים הכולל (קבוע עבור ויקיפדיה האנגלית)
    N = 6348910

    # 2. מעבר על כל מילה בשאילתה ושליפת רשימות פוסטינג
    for term, tf_q in query_counts.items():
        if term not in body_index.df:
          continue

        # חישוב IDF עבור המילה
        df = body_index.df[term]
        idf = math.log10(N / df)

        # משקל המילה בשאילתה
        w_t_q = tf_q * idf

        # קריאת רשימת הפוסטינג מה-Bucket/דיסק
        # הערה: המתודה read_posting_list חייבת להיות ממומשת בתוך InvertedIndex
        try:
            posting_list = body_index.read_posting_list(term, base_dir='.')
        except KeyError:
            continue

        # 3. עדכון ציונים לכל המסמכים המכילים את המילה
        for doc_id, tf_d in posting_list:
            # חישוב TF-IDF של המסמך (w_t_d)
            w_t_d = tf_d * idf
            # הוספה לצובר (המכפלה הסקלרית במונה)
            candidate_scores[doc_id] += (w_t_q * w_t_d)

    if not candidate_scores:
      return jsonify([])

    # 4. נרמול ומיון (נרמול לפי אורך המסמך למניעת הטיה)
    # הערה: אם יש לך מילון של נורמות שהכנת ב-GCP (למשל doc_norms), השתמש בו כאן.
    # אם אין, הדירוג יתבסס על המכפלה הסקלרית בלבד.
    results = []
    for doc_id, score in candidate_scores.items():
        try:
            doc_id = int(doc_id)
        except ValueError:
            raise ValueError(f"Document ID {doc_id} is not an integer.")

        norm = doc_norms.get(doc_id, 1)
        final_score = score / norm
        results.append((doc_id, final_score))

    # מיון לפי הציון מהגבוה לנמוך ולקיחת 100 הראשונים
    top_100 = sorted(results, key=lambda x: x[1], reverse=True)[:100]

    # 5. המרה לפורמט הנדרש: (wiki_id, title)
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
    # 1. טוקניזציה של השאילתה - שימוש בטוקנייזר שסופק
    #
    query_tokens = tokenize_txt(query)

    # שימוש ב-Counter כדי לספור מילים ייחודיות
    # המפתח הוא doc_id והערך הוא מספר המילים השונות מהשאילתה שמופיעות בכותרת
    doc_scores = Counter()

    # 2. מעבר על כל מילה בשאילתה
    for term in query_tokens:
        # דילוג על מילים שלא נמצאות באינדקס הכותרות
        # אנו משתמשים במשתנה הגלובלי title_index שנטען ב-initialize
        if term not in title_index.df:
            continue

        # שליפת רשימת המסמכים (Posting List) עבור המילה
        # הפונקציה read_posting_list מוגדרת ב-inverted_index_gcp.py
        # היא קוראת את המידע מה-Bucket בעזרת המיקומים שנשמרו בזיכרון
        #
        try:
            posting_list = title_index.read_posting_list(term, base_dir='.')
        except Exception as e:
            print(f"Error reading posting list for term {term}: {e}")
            continue
        # עדכון הציון לכל מסמך שבו המילה מופיעה
        # אנו מוסיפים 1 עבור כל מילה ייחודית מהשאילתה שנמצאת בכותרת
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1

    # 3. מיון התוצאות
    # המיון הוא לפי הציון (מספר המילים המותאמות) בסדר יורד
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # 4. יצירת רשימת התוצאות בפורמט (doc_id, title)
    # שימוש במשתנה הגלובלי id_to_title לתרגום ID לכותרת
    #
    for doc_id, score in sorted_docs:
        title = id_to_title.get(doc_id, "Title Unknown")
        res.append((str(doc_id), title))

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
    # 1. טוקניזציה של השאילתה
    #
    query_tokens = tokenize_txt(query)

    # מונה לציונים: {doc_id: unique_match_count}
    doc_scores = Counter()

    # 2. מעבר על כל מילה בשאילתה
    for term in query_tokens:
        # בדיקה אם המילה קיימת באינדקס העוגנים
        # אנו מניחים ש-anchor_index נטען ב-initialize בצורה דומה ל-title_index
        #
        if term not in anchor_index.df:
            continue

        # שליפת רשימת הפוסטינג עבור המילה מתוך ה-Bucket
        # הפונקציה קוראת את המידע הבינארי וממירה אותו לרשימת (doc_id, tf)
        #
        try:
            posting_list = anchor_index.read_posting_list(term, base_dir='.')
        except Exception as e:
            print(f"Error reading posting list for term {term}: {e}")
            continue

        # עדכון הציון: +1 לכל מסמך שמקושר עם המילה הזו בטקסט העוגן
        # ה-tf כאן מייצג כמה פעמים המילה הופיעה בקישורים לאותו דף, אך הדירוג הבינארי מתעלם מכך
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1

    # 3. מיון התוצאות
    # המיון הוא לפי הציון (מספר המילים הייחודיות) בסדר יורד
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # 4. המרה לפורמט (doc_id, title)
    # שימוש ב-id_to_title לתרגום מזהה מסמך לכותרת
    #
    for doc_id, score in sorted_docs:
        title = id_to_title.get(doc_id, "Title Unknown")
        res.append((str(doc_id), title))

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
    # מעבר על רשימת ה-IDs שהתקבלו ב-Body של הבקשה
    for doc_id in wiki_ids:
        # שליפת הציון מהמילון הגלובלי pagerank שנטען ב-initialize
        # אם ה-ID לא קיים במילון (למשל דף ללא קישורים נכנסים או דף שלא חושב), נחזיר 0.0
        try:
            # המרה ל-int ליתר ביטחון, למקרה שה-JSON העביר מחרוזות
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
    # מעבר על רשימת ה-IDs שהתקבלו מהמשתמש
    for doc_id in wiki_ids:
        # שליפת הערך מהמילון הגלובלי pageviews
        # אנו משתמשים ב-.get כדי להחזיר 0 אם הדף לא נמצא במאגר
        try:
            # המרה ל-int למקרה שהקלט הגיע כמחרוזות
            val = pageviews.get(int(doc_id), 0)
        except (ValueError, TypeError):
            val = 0

        res.append(val)
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
