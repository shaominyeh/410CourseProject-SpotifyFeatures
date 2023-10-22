import metapy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import preprocess

def preprocess_tasks():
    preprocess.extract_columns()
    preprocess.make_corpus()
    return pd.read_csv('../data/formatted_songs.csv')

songs = preprocess_tasks()

tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
tok = metapy.analyzers.LowercaseFilter(tok)
tok = metapy.analyzers.Porter2Filter(tok)
# tok = metapy.analyzers.ListFilter(tok, "../config/stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)
filtered_lyrics = []
for text in songs['lyrics']:
    tok.set_content(text.strip()) 
    tokens = [token for token in tok]
    filtered_lyrics.append(' '.join(tokens))
songs['lyrics'] = filtered_lyrics

X = filtered_lyrics
y = songs['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

tfidf_model = TfidfVectorizer()
X_train = tfidf_model.fit_transform(X_train)
X_test = tfidf_model.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)

query = songs.iloc[2]['lyrics']
# query = "test"
new_query_vector = tfidf_model.transform([query])
predicted_score = model.predict(new_query_vector)
print(predicted_score)