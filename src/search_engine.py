"""This module contains the TF-IDF search engine."""
import metapy
import numpy as np
import pandas as pd

import preprocess

def preprocess_tasks():
    """Preprocesses csv and data files."""
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('../data/formatted_songs.csv')

def load_ranker():
    """BM25 ranking function."""
    return metapy.index.OkapiBM25(k1=1.25,b=0.6,k3=0.0)

def ranking(songs, user_query, config):
    """Scores the most similar songs."""
    cfg = config
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker()

    query_name = user_query
    query = metapy.index.Document()
    query.content(query_name.lower())

    top_k = 200 # High value to combat duplicated songs
    results = ranker.score(idx, query, top_k)

    songs_list = np.zeros(len(songs))
    for result in results: # Result is formatted as (song index, score)
        songs_list[result[0]] = result[1]
    return songs_list

def separated_ranking(songs, user_query, title_weight, artist_weight, lyrics_weight):
    """Separated ranking to separate title, artist, and lyrics."""
    return title_weight * ranking(songs, user_query, "../config/config_title.toml")\
          + artist_weight * ranking(songs, user_query, "../config/config_artist.toml")\
          + lyrics_weight * ranking(songs, user_query, "../config/config_lyrics.toml")

def results_list(songs, songs_list, top_k):
    """Computes the top_k songs."""
    count = 0
    top_set = set()
    top_songs = []
    songs_indices = np.argsort(songs_list)[::-1] # Sort to find highest scoring indices
    for song in songs_indices:
        if songs_list[song] <= 0 or count >= top_k: # No songs left or met song count
            break
        song_pair = (songs.iloc[song]['track_name'], songs.iloc[song]['track_artist'])
        if song_pair not in top_set:
            top_songs.append((song_pair, songs_list[song])) # Adding the score as well
            top_set.add(song_pair)
            count += 1
    return top_songs

def print_results(top_songs):
    """Prints the top songs."""
    if len(top_songs) == 0:
        print("No Songs Returned")
    else:
        print("The top 10 songs for your query are: ")
        for pair, score in top_songs:
            print("Song Title: {} | Artist: {} | Score {}".format(pair[0], pair[1], score))

def params_test(songs, queries, params_file):
    file_name = "../data/queries/" + params_file
    for query in queries:
        songs_list = separated_ranking(songs, query, 0.33, 0.33, 0.33)
        top_songs = results_list(songs, songs_list, 10)
        with open(file_name, 'a') as file:
            file.write("Query is: {}\n".format(query))
            for pair, _ in top_songs:
                file.write("{} by: {}\n".format(pair[0], pair[1]))
            file.write("\n")

def query_search(query, rank_separated, title_weight, artist_weight, lyrics_weight):
    """Accessible method for web app."""
    songs = preprocess_tasks()

    if rank_separated:
        songs_list = separated_ranking(songs, query, title_weight,\
                                                       artist_weight, lyrics_weight)
    else:
        songs_list = ranking(songs, query, "../config/config.toml")
    return results_list(songs, songs_list, 50)

if __name__ == '__main__':
    songs = preprocess_tasks()
    IS_PARAM_TEST = False # Boolean, use if you want to test parameters only (False for general use).

    if not IS_PARAM_TEST:
        USER_QUERY = "hello" # Any ASCII String
        IS_SEPARATED_RANK = True # Boolean
        if IS_SEPARATED_RANK:
            songs_list = separated_ranking(songs, USER_QUERY, 0.33, 0.33, 0.33)
        else:
            songs_list = ranking(songs, USER_QUERY, "../config/config.toml")

        print_results(results_list(songs, songs_list, 10))
    else:
        # If you want to test out parameters test, change them and call params_test to a new file.
        QUERY_LIST = ["hello", "bye", "weeknd", "cold", "the", "polo", "mars", "hey", "no", "poker"]
        PARAMS_NAME = "k1_125__b_06.txt"
        params_test(songs, QUERY_LIST, PARAMS_NAME)
