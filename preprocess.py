import csv
import metapy
import os
import pandas as pd

def extract_columns():
  if not os.path.isfile('formatted_songs.csv'):
    if os.path.isfile('songs/songs.dat'):
      os.remove('songs/songs.dat')
    songs = pd.read_csv('spotify_songs.csv')
    songs = songs.dropna()
    songs = songs.drop_duplicates()
    songs = songs[songs['language'] == 'en']
    songs = songs.drop(['track_album_id', 'playlist_id', 'language'], axis = 1)
    songs.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
    songs.to_csv('formatted_songs.csv')
  return 'formatted_songs.csv'

def make_corpus():
  if not os.path.isfile('songs/songs.dat'):
    csv_file = 'formatted_songs.csv'
    output_file = 'songs/songs.dat'
    csv_reader = csv.reader(open(csv_file, encoding='utf-8'))
    next(csv_reader, None)
    with open(output_file, 'w') as dat_file:
      for row in csv_reader:
        combined_content = "{} {} {}".format(row[2], row[3], row[4])
        dat_file.write(combined_content.lower() + "\n") 