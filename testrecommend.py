import csv
import math
import metapy
import numpy as np
import pandas as pd
import sys

import preprocess

def preprocess_tasks():
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('data/formatted_songs.csv')

def load_ranker(cfg_file):
    return metapy.index.OkapiBM25(k1=1.66,b=0.72,k3=2.2) 

def combined_ranking(songs, user_query, config):
    cfg = config
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)

    query_name = user_query
    query = metapy.index.Document()
    query.content(query_name.lower())

    results = ranker.score(idx, query)

    songs_list = np.zeros(len(songs))
    for result in results:
        songs_list[result[0]] += result[1]
    return songs_list

# def separated_ranking(songs, user_query, title_weight, artist_weight, lyrics_weight):

#     return 

def print_results(songs_list, top_k):
    if (len(songs_list) == 0):
        print("No Songs Returned")
    else:
        count = 0
        topset = set()
        songs_list = np.argsort(songs_list)[::-1]
        for song in songs_list:
            if count >= top_k:
                break
            songpair = (songs.iloc[song]['track_name'], songs.iloc[song]['track_artist'])
            if songpair not in topset:
                print(songpair)
                topset.add(songpair)
                count += 1

if __name__ == '__main__':
    songs = preprocess_tasks()
    query = "love"

    songs_list = combined_ranking(songs, query, "config/config.toml")
    # songs_list = separated_ranking(songs, query, 0.3, 0.3, 0.4)

    print_results(songs_list, 10)
