import csv
import math
import metapy
import pandas as pd
import sys

import preprocess

def load_ranker(cfg_file):
    return metapy.index.OkapiBM25(k1=1.66,b=0.72,k3=2.2) 

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    preprocess.extract_columns()
    preprocess.make_corpus()
    songs = pd.read_csv('formatted_songs.csv')

    cfg = sys.argv[1]
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)

    query_name = "Russian Roulette"
    query = metapy.index.Document()
    query.content(query_name.lower())

    top_k = 10
    results = ranker.score(idx, query, top_k)

    for result in results:
        print(songs.iloc[result[0]]['track_name'])
