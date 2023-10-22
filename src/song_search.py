import pandas as pd

import preprocess

def preprocess_tasks():
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('../data/formatted_songs.csv')

def nonduplicated_dict(songs):
    song_dict = {}
    for song in range(len(songs)):
        song_pair = (songs.iloc[song]['track_name'], songs.iloc[song]['track_artist'])
        if song_pair not in song_dict:
            song_dict[song_pair] = song
    return song_dict

def song_search(query_name, is_title):
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
    query_name = "tHE"
    is_title = False

    print(song_search(query_name, is_title))
