"""This module contains the Similar Song Search."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

import preprocess

def preprocess_tasks():
    """Preprocesses csv and data files."""
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('../data/formatted_songs.csv')

def tfidf_features(song_index, songs):
    """Computes similar songs with just TF-IDF."""
    lyrics = songs['lyrics']
    tfidf_model = TfidfVectorizer()
    tfidf_features = tfidf_model.fit_transform(lyrics)
    features = tfidf_features.toarray()

    knn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
    knn_model.fit(features)

    query_features = features[song_index]
    return knn_model.kneighbors(query_features.reshape(1, -1))

def music_features(song_index, songs):
    """Computes similar songs with just musical features."""
    features = songs[['track_popularity','danceability','energy','key','loudness','mode',\
                      'speechiness','acousticness','instrumentalness','liveness',\
                        'valence','tempo','duration_ms']]

    knn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
    knn_model.fit(features)

    query_features = features.iloc[song_index].values.reshape(1, -1)
    return knn_model.kneighbors(query_features.reshape(1, -1))

def combined_features(song_index, songs):
    """Computes similar songs with both features."""
    lyrics = songs['lyrics']
    tfidf_model = TfidfVectorizer()
    tfidf_features = tfidf_model.fit_transform(lyrics)
    musical_features = songs[['track_popularity','danceability','energy','key','loudness',\
                              'mode','speechiness','acousticness','instrumentalness',\
                                'liveness','valence','tempo','duration_ms']]
    features = hstack([tfidf_features.toarray(), musical_features])

    knn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
    knn_model.fit(features)

    query_features = features.getrow(song_index).toarray()
    return knn_model.kneighbors(query_features)

def similar_songs(user_index, selected_feature):
    """Accessible method for web app."""
    songs = preprocess_tasks()

    if selected_feature == "tfidf":
        distances, indices = tfidf_features(user_index, songs)
    elif selected_feature == "musical":
        distances, indices = music_features(user_index, songs)
    else:
        distances, indices = combined_features(user_index, songs)
    return results_list(songs, distances, indices)

def results_list(songs, distances, indices):
    """Computes the 25 closest songs."""
    count = 0
    top_set = set()
    top_songs = []
    top_k = 25
    for song in indices[0]:
        if count >= top_k:
            break
        song_pair = (songs.iloc[song]['track_name'], songs.iloc[song]['track_artist'])
        if song_pair not in top_set:
            top_songs.append((song_pair,distances[0][count]))
            top_set.add(song_pair)
            count += 1
    return top_songs

def print_results(top_songs):
    """Prints the top songs."""
    if len(top_songs) == 0:
        print("No Songs Returned")
    else:
        print("The closest 25 songs for your song are: ")
        for pair, score in top_songs:
            print("Song Title: {} | Artist: {} | Distance {}".format(pair[0], pair[1], score))

if __name__ == '__main__':
    CHOSEN_INDEX = 0
    WHICH_FEATURE = "tfidf"

    print_results(similar_songs(CHOSEN_INDEX, WHICH_FEATURE))
