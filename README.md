# Wikipedia Search Engine - Information Retrieval Project

An advanced search engine designed for the entire English Wikipedia corpus (~6.3 million documents), developed as part of the Information Retrieval course at Ben-Gurion University.

## 👥 Project Team
* **Eliya Ohayon**
* **Omer Dweck**
* **Ido Aloya**

## 📝 Overview
This project implements a high-performance search engine using **Python** and **Flask**, deployed on **Google Cloud Platform (GCP)**. The system is optimized to handle millions of documents with a focus on high precision, recall, and low-latency retrieval (averaging around 1 second per query).



## 🛠 Architecture & Technologies
* **Backend:** Python (Flask)
* **Infrastructure:** Google Cloud Compute Engine (`e2-standard-4`), Google Cloud Storage (GCS)
* **Processing:** NLTK, PorterStemmer, Custom Tokenizer
* **Ranking:** BM25, Cosine Similarity, PageRank, PageViews
* **Reranking:** SBERT (`multi-qa-MiniLM-L6-cos-v1`)

## 🚀 Key Features

### 1. Advanced Indexing Strategy
To ensure rapid retrieval across the 6.3M document corpus, we implemented:
* **Tiered Body Index:** A prioritized `tier1_index` containing the most relevant posting lists. The system falls back to the full index only when necessary, maintaining high speed.
* **Title & Anchor Indices:** Specialized indices that prioritize results based on page titles and internal link anchor text.

### 2. Optimized Ranking Pipeline
* **Hybrid Scoring:** Combines textual relevance (Body, Title, Anchor) with static authority metrics (PageRank & PageViews).
* **Grid Search Optimization:** We conducted a systematic **Grid Search** to determine the optimal weights for each component, maximizing the overall retrieval quality (MAP and Precision@10).
* **Semantic Reranking:** The top candidates are reranked using an SBERT model to capture deeper semantic meaning.

### 3. Experimental Extensions
* **Query Expansion:** We experimented with **WordNet** and **word2vec** for query expansion. While these methods aimed to improve recall, our testing showed that the marginal performance gains did not justify the significant increase in latency. For production efficiency, these were excluded from the final pipeline.

## 📊 Performance
* **Latency:** ~(0.4 - 1.5) seconds per query.
* **Scalability:** Built to run on a single GCP instance while efficiently querying the full Wikipedia dataset.

## 💻 Setup & Execution
The engine requires pre-computed indices to be loaded from GCS. To run the server:
```bash
python search_frontend.py
