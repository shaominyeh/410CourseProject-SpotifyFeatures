import os
import pandas as pd

def preprocess():
  if not os.path.isfile('formatted_songs.csv'):
    songs = pd.read_csv('spotify_songs.csv')
    songs = songs.dropna()
    songs = songs[songs['language'] == 'en']
    songs = songs.drop(['track_id', 'track_album_id', 'playlist_id', 'language'], axis = 1)
    songs.to_csv('formatted_songs.csv')
  return 'formatted_songs.csv'

preprocess()