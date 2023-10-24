"""This module contains preprocessing utility for datasets."""
import csv
import os
import pandas as pd

SPOTIFY_SONGS_PATH = '../data/spotify_songs.csv'
FORMATTED_SONGS_PATH = '../data/formatted_songs.csv'
SONGS_DATA_PATH = '../config/songs/songs.dat'
LYRICS_DATA_PATH = '../config/lyrics/lyrics.dat'
ARTIST_DATA_PATH = '../config/artist/artist.dat'
TITLE_DATA_PATH = '../config/title/title.dat'

def extract_columns():
    """Extracts unnecessary columns from dataset."""
    if not os.path.isfile(FORMATTED_SONGS_PATH):
        if os.path.isfile(SONGS_DATA_PATH):
            os.remove(SONGS_DATA_PATH)
        if os.path.isfile(LYRICS_DATA_PATH):
            os.remove(LYRICS_DATA_PATH)
            os.remove(ARTIST_DATA_PATH)
            os.remove(TITLE_DATA_PATH)
        songs = pd.read_csv(SPOTIFY_SONGS_PATH)
        songs = songs.dropna()
        songs = songs[songs['language'] == 'en']
        songs = songs.drop(['track_album_id', 'playlist_id', 'language'], axis = 1)
        songs.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        songs.to_csv(FORMATTED_SONGS_PATH)
    return FORMATTED_SONGS_PATH

def create_dat_file(output_file):
    """Creates the dat files for separated and combined features."""
    csv_file = FORMATTED_SONGS_PATH
    csv_reader = csv.reader(open(csv_file, encoding='utf-8'))
    next(csv_reader, None)
    with open(output_file, 'w') as dat_file:
        for row in csv_reader:
            if output_file == LYRICS_DATA_PATH:
                combined_content = "{}".format(row[4])
            elif output_file == ARTIST_DATA_PATH:
                combined_content = "{}".format(row[3])
            elif output_file == TITLE_DATA_PATH:
                combined_content = "{}".format(row[2])
            else:
                combined_content = "{} {} {}".format(row[2], row[3], row[4])
            dat_file.write(combined_content.lower() + "\n")

def make_corpus():
    """Makes corpus for separated and combined features."""
    if not os.path.isfile(SONGS_DATA_PATH) or not os.path.isfile(LYRICS_DATA_PATH):
        create_dat_file(SONGS_DATA_PATH)
        create_dat_file(LYRICS_DATA_PATH)
        create_dat_file(ARTIST_DATA_PATH)
        create_dat_file(TITLE_DATA_PATH)
