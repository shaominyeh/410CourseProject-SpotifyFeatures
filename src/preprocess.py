import csv
import metapy
import os
import pandas as pd

spotify_songs_path = '../data/spotify_songs.csv'
formatted_songs_path = '../data/formatted_songs.csv'
songs_data_path = '../config/songs/songs.dat'
lyrics_data_path = '../config/lyrics/lyrics.dat'
artist_data_path = '../config/artist/artist.dat'
title_data_path = '../config/title/title.dat'

def extract_columns():
    if not os.path.isfile(formatted_songs_path):
        if os.path.isfile(songs_data_path): os.remove(songs_data_path)
        if os.path.isfile(lyrics_data_path):
            os.remove(lyrics_data_path)
            os.remove(artist_data_path)
            os.remove(title_data_path)
        songs = pd.read_csv(spotify_songs_path)
        songs = songs.dropna()
        songs = songs[songs['language'] == 'en']
        songs = songs.drop(['track_album_id', 'playlist_id', 'language'], axis = 1)
        songs.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        songs.to_csv(formatted_songs_path)
    return formatted_songs_path

def create_dat_file(output_file):
    csv_file = formatted_songs_path
    csv_reader = csv.reader(open(csv_file, encoding='utf-8'))
    next(csv_reader, None)
    with open(output_file, 'w') as dat_file:
        for row in csv_reader:
            if output_file == lyrics_data_path: combined_content = "{}".format(row[4])
            elif output_file == artist_data_path: combined_content = "{}".format(row[3])
            elif output_file == title_data_path: combined_content = "{}".format(row[2])
            else: combined_content = "{} {} {}".format(row[2], row[3], row[4])
            dat_file.write(combined_content.lower() + "\n")  

def make_corpus():
    if not os.path.isfile(songs_data_path) or not os.path.isfile(lyrics_data_path):
        create_dat_file(songs_data_path)
        create_dat_file(lyrics_data_path)
        create_dat_file(artist_data_path)
        create_dat_file(title_data_path)
