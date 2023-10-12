import csv
import math
import metapy
import pandas as pd
import sys

import preprocess

def load_ranker(cfg_file):
    return metapy.index.OkapiBM25(k1=1.66,b=0.72,k3=2.2) 

if __name__ == '__main__':
    preprocess.extract_columns()
    preprocess.make_corpus()
    songs = pd.read_csv('formatted_songs.csv')

    cfg = "config.toml"
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)

    query_name = "i just wanna rock"
    query = metapy.index.Document()
    query.content(query_name.lower())

    top_k = len(songs)
    results = ranker.score(idx, query, top_k)

    count = 0
    topset = set()
    for result in results:
        if count >= 10:
            break
        songpair = (songs.iloc[result[0]]['track_name'], songs.iloc[result[0]]['track_artist'])
        if songpair not in topset:
            print(songpair)
            topset.add(songpair)
            count += 1
