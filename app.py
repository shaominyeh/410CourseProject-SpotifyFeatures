from flask import Flask, request, render_template, redirect, url_for

import search_engine

app = Flask(__name__, template_folder='templates')

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
    print(top_songs)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=32000, threaded=False)