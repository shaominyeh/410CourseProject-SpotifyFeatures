from flask import Flask, request, render_template

import search_engine
import similar_songs
import song_search

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def home_page():
    return render_template('home-page.html')

@app.route('/search_engine', methods=['POST'])
def search_query():
    return render_template('search-engine-query.html')

@app.route('/search_engine_list', methods=['POST'])
def search_engine_list():
    if request.form['title_weight'] and request.form['artist_weight'] and request.form['lyrics_weight']: 
        top_songs = search_engine.query_search(request.form['query'], True, float(request.form['title_weight']),\
                                    float(request.form['artist_weight']), float(request.form['lyrics_weight']))
    else: 
        top_songs = search_engine.query_search(request.form['query'], False, 0.0, 0.0, 0.0)
    return render_template('search-engine-results.html', songs=top_songs)

@app.route('/song_search_similar', methods=['POST'])
def song_search_similar():
    return render_template('song-search-similar.html')

@app.route('/song_search_similar_list', methods=['POST'])
def song_search_similar_list():
    all_songs = song_search.song_search(request.form['query'], \
                                        True if request.form['search_choice'] == "title" else False)
    return render_template('song-search-similar-list.html', songs=all_songs)

@app.route('/similar_songs_query/<user_index>', methods=['POST'])
def similar_songs_query(user_index):
    top_songs = similar_songs.similar_songs(int(user_index), request.form['selected_feature'])
    return render_template('similar-songs-results.html', songs=top_songs)

if __name__ == '__main__':
    app.run(threaded=False)