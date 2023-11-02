"""This module contains the Sentiment Analysis Model."""
import metapy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import preprocess

def preprocess_tasks():
    """Preprocesses csv and data files."""
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('../data/formatted_songs.csv')

def tokenize_query(songs):
    """Tokenizes the entire songs dataset."""
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.Porter2Filter(tok)
    # tok = metapy.analyzers.ListFilter(tok, "../config/stopwords.txt", \
                                    #   metapy.analyzers.ListFilter.Type.Reject)
    filtered_lyrics = []
    for text in songs['lyrics']:
        tok.set_content(text.strip())
        tokens = [token for token in tok]
        filtered_lyrics.append(' '.join(tokens))
    return filtered_lyrics

def train_model(X, y, random_state):
    """Trains the model using TF-IDF and regression."""
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=random_state)

    tfidf_model = TfidfVectorizer()
    X_train = tfidf_model.fit_transform(X_train)
    X_test = tfidf_model.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return tfidf_model, model, mse

def compute_sentiment(songs, user_index, model, tfidf_model):
    """Computing sentiment of the chosen index."""
    query = songs.iloc[user_index]['lyrics']
    query_vector = tfidf_model.transform([query])
    predicted_score = model.predict(query_vector)
    return predicted_score[0], songs.iloc[user_index]['valence'], query_vector

def compute_words(tfidf_model, query_vector):
    """Computes the most significant words in the index."""
    feature_names = tfidf_model.get_feature_names()

    word_dict = {}
    for index, score in zip(query_vector[0].indices, query_vector[0].data):
        word_dict[index] = score

    sorted_scores = sorted(word_dict.items(), key=lambda x:x[1], reverse = True)
    return sorted_scores[:10], feature_names

def result_items(chosen_index, unified_state, is_tokenized_query):
    songs = preprocess_tasks()
    if is_tokenized_query:
        X = tokenize_query(songs)
    else:
        X = songs['lyrics']
    y = songs['valence']

    user_tfidf_model, user_model, calculated_mse = train_model(X, y, unified_state)
    model_predicted_score, song_actual_score, new_query_vector = compute_sentiment(\
        songs, chosen_index, user_model, user_tfidf_model)
    song_top_words, model_feature_names = compute_words(user_tfidf_model, new_query_vector)
    return model_predicted_score, song_actual_score,\
                 song_top_words, model_feature_names, calculated_mse

def print_items(predicted_score, actual_score, top_words, feature_names, mse):
    """Prints scores and significant words."""
    print("Calculated MSE: {}".format(mse))
    print("Predicted Score: {} | Actual Score: {}".format(predicted_score, actual_score))

    print("The most significant words in your song are: ")
    for index, score in top_words:
        print(feature_names[index], score)

if __name__ == '__main__':
    CHOSEN_INDEX = 0
    UNIFIED_STATE = 47
    IS_TOKENIZED_QUERY = False

    main_predicted_score, main_actual_score, main_top_words, main_feature_names, main_mse\
          = result_items(CHOSEN_INDEX, UNIFIED_STATE, IS_TOKENIZED_QUERY)
    print_items(main_predicted_score, main_actual_score,\
                 main_top_words, main_feature_names, main_mse)
