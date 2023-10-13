import csv
import metapy
import os
import pandas as pd

def extract_columns():
    if not os.path.isfile('data/formatted_songs.csv'):
        if os.path.isfile('config/songs/songs.dat'): os.remove('config/songs/songs.dat')
        if os.path.isfile('config/lyrics/lyrics.dat'):
            os.remove('config/lyrics/lyrics.dat')
            os.remove('config/artist/artist.dat')
            os.remove('config/title/title.dat')
        songs = pd.read_csv('data/spotify_songs.csv')
        songs = songs.dropna()
        songs = songs[songs['language'] == 'en']
        songs = songs.drop(['track_album_id', 'playlist_id', 'language'], axis = 1)
        songs.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        songs.to_csv('data/formatted_songs.csv')
    return 'data/formatted_songs.csv'

def create_dat_file(output_file):
    csv_file = 'data/formatted_songs.csv'
    csv_reader = csv.reader(open(csv_file, encoding='utf-8'))
    next(csv_reader, None)
    with open(output_file, 'w') as dat_file:
        for row in csv_reader:
            if output_file == 'config/lyrics/lyrics.dat': combined_content = "{}".format(row[4])
            elif output_file == 'config/artist/artist.dat': combined_content = "{}".format(row[3])
            elif output_file == 'config/title/title.dat': combined_content = "{}".format(row[2])
            else: combined_content = "{} {} {}".format(row[2], row[3], row[4])
            dat_file.write(combined_content.lower() + "\n")  

def make_corpus():
    if not os.path.isfile('config/songs/songs.dat') or not os.path.isfile('config/lyrics/lyrics.dat'):
        create_dat_file('config/songs/songs.dat')
        create_dat_file('config/lyrics/lyrics.dat')
        create_dat_file('config/artist/artist.dat')
        create_dat_file('config/title/title.dat')
