import metapy
import math
import pandas as pd

import preprocess

preprocess.preprocess()

documents = [
    {"id": "doc1", "content": "This is the first document. It is about BM25 ranking.", "title": "Document One", "author": "John Doe"},
    {"id": "doc2", "content": "The second document is also about BM25 and ranking.", "title": "Document Two", "author": "Jane Smith"},
    {"id": "doc3", "content": "Document three discusses the BM25 formula in detail.", "title": "Document Three", "author": "Alice Johnson"},
]

# Create a forward index for the documents
idx = metapy.index.make_inverted_index('config.toml')

# Define a metapy ranker with BM25 parameters
ranker = metapy.index.OkapiBM25(k1=1.5, b=0.75)

# Preprocess and tokenize the query
query = "BM25 ranking formula"
tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
tok = metapy.analyzers.LowercaseFilter(tok)
tok.set_content(query)
tokens = [token for token in tok]

# Retrieve and score documents using BM25
top_k = 10  # Adjust this for the number of top documents you want
results = ranker.score(idx, tokens, num_results=top_k)

# Print the ranked documents with titles and authors
for result in results:
    doc_id = result[0]
    doc = next((d for d in documents if d['id'] == doc_id), None)
    if doc:
      print("Document '{}' by {}: Score = {:.4f}".format(doc['title'], doc['author'], result[1]))

# documents = [
#     {"title": "Document 1", "author": "Author 1", "text": "This is the first document. It is about BM25 ranking."},
#     {"title": "Document 2", "author": "Author 2", "text": "The second document is also about BM25 and ranking."},
#     {"title": "Document 3", "author": "Author 3", "text": "Document three discusses the BM25 formula in detail."},
# ]

# # Create a Pandas DataFrame to store documents
# df = pd.DataFrame(documents)

# # Preprocessing and tokenization
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     return text.split()

# # Calculate inverse document frequency (IDF) for terms using Pandas
# def calculate_idf(df, field):
#     idf = {}
#     total_docs = len(df)
#     for i, row in df.iterrows():
#         terms = preprocess(row[field])
#         unique_terms = set(terms)
#         for term in unique_terms:
#             if term in idf:
#                 idf[term] += 1
#             else:
#                 idf[term] = 1
#     for term, freq in idf.items():
#         idf[term] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)
#     return idf

# idf_text = calculate_idf(df, 'text')
# idf_author = calculate_idf(df, 'author')
# idf_title = calculate_idf(df, 'title')

# # Calculate term frequency (TF) for a query for each field
# def calculate_tf(query, field):
#     tf = {}
#     terms = preprocess(query[field])
#     for term in terms:
#         if term in tf:
#             tf[term] += 1
#         else:
#             tf[term] = 1
#     return tf

# # BM25 ranking
# def bm25_ranking(query, df, idf_text, idf_author, idf_title, k1=1.5, b=0.75):
#     scores = {}
#     tf_query_text = calculate_tf(query, 'text')
#     tf_query_author = calculate_tf(query, 'author')
#     tf_query_title = calculate_tf(query, 'title')
#     avg_doc_length_text = df['text'].apply(lambda x: len(preprocess(x))).mean()
#     avg_doc_length_author = df['author'].apply(lambda x: len(preprocess(x))).mean()
#     avg_doc_length_title = df['title'].apply(lambda x: len(preprocess(x))).mean()

#     for i, row in df.iterrows():
#         terms_text = preprocess(row['text'])
#         doc_length_text = len(terms_text)
#         score = 0.0

#         for term in tf_query_text:
#             if term in idf_text:
#                 df = idf_text[term]
#                 tf = terms_text.count(term)
#                 numerator = (tf * (k1 + 1))
#                 denominator = (tf + k1 * (1 - b + b * doc_length_text / avg_doc_length_text))
#                 score += df * (numerator / denominator) * (1 + k1) * tf_query_text[term]

#         # You can repeat the similar process for author and title
#         terms_author = preprocess(row['author'])
#         doc_length_author = len(terms_author)
#         for term in tf_query_author:
#             if term in idf_author:
#                 df = idf_author[term]
#                 tf = terms_author.count(term)
#                 numerator = (tf * (k1 + 1))
#                 denominator = (tf + k1 * (1 - b + b * doc_length_author / avg_doc_length_author))
#                 score += df * (numerator / denominator) * (1 + k1) * tf_query_author[term]

#         terms_title = preprocess(row['title'])
#         doc_length_title = len(terms_title)
#         for term in tf_query_title:
#             if term in idf_title:
#                 df = idf_title[term]
#                 tf = terms_title.count(term)
#                 numerator = (tf * (k1 + 1))
#                 denominator = (tf + k1 * (1 - b + b * doc_length_title / avg_doc_length_title))
#                 score += df * (numerator / denominator) * (1 + k1) * tf_query_title[term]

#         scores[i] = score

#     # Sort documents by score in descending order
#     sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#     return sorted_docs

# query = {"title": "BM25 ranking formula", "author": "Author 1", "text": "relevant keywords for ranking BM25 formula"}
# ranked_docs = bm25_ranking(query, df, idf_text, idf_author, idf_title)

# print("BM25 Ranking Results for Query:")
# for i, (doc_idx, score) in enumerate(ranked_docs):
#     doc_info = df.loc[doc_idx]
#     print("Rank {} - Document '{}' by {}: Score = {:.4f}".format(i + 1, doc_info['title'], doc_info['author'], score))
