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

def ranking(songs, user_query, config):
    cfg = config
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)

    query_name = user_query
    query = metapy.index.Document()
    query.content(query_name.lower())

    results = ranker.score(idx, query)

    songs_list = np.zeros(len(songs))
    for result in results:
        songs_list[result[0]] = result[1]
    return songs_list

def separated_ranking(songs, user_query, title_weight, artist_weight, lyrics_weight):
    return title_weight * ranking(songs, user_query, "config/config_title.toml")\
          + artist_weight * ranking(songs, user_query, "config/config_artist.toml")\
          + lyrics_weight * ranking(songs, user_query, "config/config_lyrics.toml")

def print_results(songs_list, top_k):
    if (len(songs_list) == 0):
        print("No Songs Returned")
    else:
        count = 0
        topset = set()
        songs_indices = np.argsort(songs_list)[::-1]
        printed_one = False
        for song in songs_indices:
            if songs_list[song] <= 0 or count >= top_k:
                if printed_one == False:
                    print("No Songs Returned")
                break
            songpair = (songs.iloc[song]['track_name'], songs.iloc[song]['track_artist'])
            if songpair not in topset:
                printed_one = True
                print(songpair)
                topset.add(songpair)
                count += 1

if __name__ == '__main__':
    songs = preprocess_tasks()
    query = "the"

    rank_separate = True
    if rank_separate: songs_list = separated_ranking(songs, query, 0.3, 0.4, 0.3)
    else: songs_list = ranking(songs, query, "config/config.toml")

    print_results(songs_list, 10)
