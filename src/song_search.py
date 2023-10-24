"""This module searches for a song index."""
import pandas as pd

import preprocess

def preprocess_tasks():
    """Preprocesses csv and data files."""
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('../data/formatted_songs.csv')

def nonduplicated_dict(songs):
    """Removes duplicates from songs data."""
    song_dict = {}
    for song in range(len(songs)):
        song_pair = (songs.iloc[song]['track_name'], songs.iloc[song]['track_artist'])
        if song_pair not in song_dict:
            song_dict[song_pair] = song
    return song_dict

def song_search(query_name, is_title):
    """Searches for song based on given query and search mode."""
    songs = preprocess_tasks()
    song_dict = nonduplicated_dict(songs)

    query_name = query_name.lower()
    song_list = []
    for (song, index) in song_dict.items():
        if query_name in song[0].lower() and is_title or \
            query_name in song[1].lower() and not is_title:
            song_list.append((song,index))
    return song_list

if __name__ == '__main__':
    USER_QUERY = "tHE"
    IS_TITLE_SEARCH = False

    print(song_search(USER_QUERY, IS_TITLE_SEARCH))
