import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

import preprocess

def preprocess_tasks():
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('data/formatted_songs.csv')

songs = preprocess_tasks()
lyrics = songs['lyrics']
features = songs[['track_popularity','danceability','energy','key','loudness','mode','speechiness'\
                  ,'acousticness','instrumentalness','liveness','valence','tempo','duration_ms']]

chosen_index = 2
tfidf_vectorizer = TfidfVectorizer()
lyrics_tfidf = tfidf_vectorizer.fit_transform(lyrics)

combined_features = hstack([lyrics_tfidf.toarray(), features])
# combined_features = features
similarities = cosine_similarity(combined_features)

knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
knn_model.fit(combined_features)

# query_features = combined_features[chosen_index]
# # query_features = combined_features.iloc[chosen_index].values.reshape(1, -1)

# distances, indices = knn_model.kneighbors(query_features.reshape(1, -1))
query_features = combined_features.getrow(chosen_index).toarray()

distances, indices = knn_model.kneighbors(query_features)

count = 0
topset = set()
top_songs = []
top_k = 10
for song in indices[0]:
    if count >= top_k:
        break
    songpair = (songs.iloc[song]['track_name'], songs.iloc[song]['track_artist'])
    if songpair not in topset:
        top_songs.append(songpair)
        topset.add(songpair)
        count += 1

print(top_songs)