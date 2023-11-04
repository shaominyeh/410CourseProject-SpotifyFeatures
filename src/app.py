"""This module contains the Web Application."""
import random
from flask import Flask, request, render_template

import search_engine
import sentiment_analysis
import similar_songs
import song_search

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def home_page():
    """Home page for the website."""
    return render_template('home-page.html')

@app.route('/search_engine', methods=['POST'])
def search_query():
    """User inputs for the search engine."""
    return render_template('search-engine-query.html')

@app.route('/search_engine_list', methods=['POST'])
def search_engine_list():
    """Search engine list based on the previous query."""
    title_weight = request.form['title_weight']
    artist_weight = request.form['artist_weight']
    lyrics_weight = request.form['lyrics_weight']
    if title_weight and artist_weight and lyrics_weight: # Separated Ranking
        top_songs = search_engine.query_search(request.form['query'], True,\
                                    float(title_weight), float(artist_weight), float(lyrics_weight))
    else: # Combined Ranking
        top_songs = search_engine.query_search(request.form['query'], False, 0.0, 0.0, 0.0)
    return render_template('search-engine-results.html', songs=top_songs)

@app.route('/song_search_similar', methods=['POST'])
def song_search_similar():
    """User inputs for the song search in the similar songs model."""
    return render_template('song-search-similar.html')

@app.route('/song_search_similar_list', methods=['POST'])
def song_search_similar_list():
    """Song search list for choosing a song in the similar songs model."""
    all_songs = song_search.song_search(\
        request.form['query'], bool(request.form['search_choice'] == "title"))
    return render_template('song-search-similar-list.html', songs=all_songs)

@app.route('/similar_songs_query/<user_index>', methods=['POST'])
def similar_songs_list(user_index):
    """Similar song list based on the chosen song and feature option."""
    top_songs = similar_songs.similar_songs(int(user_index), request.form['selected_feature'])
    return render_template('similar-songs-results.html', songs=top_songs)

@app.route('/song_search_sentiment', methods=['POST'])
def song_search_sentiment():
    """User inputs for the song search in the sentiment analysis model."""
    return render_template('song-search-sentiment.html')

@app.route('/song_search_sentiment_list', methods=['POST'])
def song_search_sentiment_list():
    """Song search list for choosing a song in the sentiment analysis model."""
    all_songs = song_search.song_search(\
        request.form['query'], bool(request.form['search_choice'] == "title"))
    return render_template('song-search-sentiment-list.html', songs=all_songs)

@app.route('/sentiment_analysis_results/<user_index>', methods=['POST'])
def sentiment_analysis_results(user_index):
    """Sentiment analysis list based on the chosen song and feature options."""
    if request.form['random_state']: # Random State Option
        random_state = int(request.form['random_state'])
    else:
        random_state = random.randint(0, 100000)
    if request.form['tokenization'] == 'True': # Tokenization Option
        tokenization = True
    else:
        tokenization = False
    predicted_score, actual_score, top_words, feature_names, mse = \
        sentiment_analysis.result_items(int(user_index), random_state, tokenization)
    return render_template('sentiment-analysis-results.html', predicted_score=predicted_score,\
            actual_score=actual_score, top_words=top_words, feature_names=feature_names, mse=mse,\
                  random_state=random_state)

if __name__ == '__main__':
    app.run(threaded=False) # Single-threaded due to Metapy and Flask conflicts
