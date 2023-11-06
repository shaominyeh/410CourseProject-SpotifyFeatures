# Spotify Features App

## CS 410 Course Project Fall 2023: Utilizing text mining and machine learning to gain insights on Spotify songs

By: Shao-Min Yeh (snyeh2@illinois.edu)

## Deliverables 

This section lists all the required deliverables for the class project:

* [Project Proposal](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/blob/main/Project%20Proposal.pdf)
* [Progress Report](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/blob/main/Project%20Progress%20Report.pdf)
    * [Repository state at the time of this report](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/2b9611861ecd8c87bafb9a20a7e786936c2bea20)
* [Code](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/src)
* Final Documentation WIP
* Presentation WIP

## Project Overview

This project consists of three different models regarding songs from a Spotify dataset all encapsulated by a Flask application. The TF-IDF Search Engine model takes a user-submitted query and outputs the highest weighted songs based on TF-IDF weighting. The Similar Songs model takes a user-submitted song and outputs the most similar songs based on TF-IDF and musical features. The Sentiment Analysis model takes a user-submitted song and outputs the sentiment analysis based on the song's lyrics. 

## Project Installation
This section consists of the following installations to run this project:

### Repository Cloning
After creating a new folder in your coding environment, simply clone the git repository:
```
git clone https://github.com/shaominyeh/410CourseProject-SpotifyFeatures.git
```

### System Environment
Metapy requires Python 3.5.6 to run (in Windows), so we will utilize [Anaconda](https://www.anaconda.com/download) to create a virtual environment.  

First, open up an Anaconda Prompt. Then, create a new Conda Environment with Python 3.5:
```
conda create -n SpotifyApp
conda activate SpotifyApp
conda install python=3.5
```
Within the repository now, select the Python Interpreter and select the created Conda environment. In VS Code, press F1 -> type "Python: Select Interpreter" -> press Python 3.5.6 ('SpotifyApp')

### Library Requirements
If the above is done correctly, all you need to do is input the following for the library requirements:
```
pip install -r requirements.txt
```
More specifically, in the command prompt (use command prompt, not terminal), it should look like the following:
```
(SpotifyApp) C:\Users\(user)>pip install -r requirements.txt
```

## Project Development
After the project installation is done, you should have a development environment containing the repository, Conda environment, and respective libraries. After this, you need to open a command prompt (if on Windows) and run the commands listed below. 

To begin, the commands in each subsection will need you to type:
```
cd src
``` 

### Flask Application
If everything is setup correctly, typing the following  and accessing http://127.0.0.1:5000/ should lead to the webpage.
```
python app.py
```
The webpage has their respective buttons and instructions regarding different options and when a user needs to input a query. Detailed pictures can be found in the final documentation and progress report. 

### Model Application
All three models can be accessed in the same way: changing the main block variables that are all caps. For example,
```python
# Change the following variables:
if __name__ == '__main__':
    CHOSEN_INDEX = 0
    UNIFIED_STATE = 47
    IS_TOKENIZED_QUERY = False
```
Then, to run a model, all you need to do is run their respective python file.
```
# Change the file to your respective model
python sentiment_analysis.py
```

## Modeling Details
This section consists of how models were implemented:

* Search Engine: This model utilizes Metapy for the TF-IDF ranking and Pandas to preprocess the dataset and catalog directories. This is done with a BM25 ranking model with Metapy's implementation. Along with this, catalog directories from the preprocessing allow for separated or combined rankings (for title/artist rankings).
* Similar Songs: This model utilizes Scikit-Learn functions (TF-IDF vectorizer and Nearest Neighbors) for ranking and Pandas for preprocessing. This is done with a KNN model with cosine-similarity. Along with this, TF-IDF is calculated with Scikit-Learn functions and musical features are calculated directly from the dataset.
* Sentiment Analysis: This model utilizes Scikit-Learn functions (Linear Regression, metrics, and Train-Test splits) for ranking functions, Metapy for query tokenization, and Pandas for preprocessing. This model utilizes the dataset's valence (emotional value) with Linear Regression to train the model and make predictions. 

## Other Important Pages

Here are the rest of the important directory paths found in this repo:

* [Sources](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/blob/main/docs/sources.txt)
* [HTML Templates](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/src/templates)
* [CSS Templates](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/src/static)
* [CSV Datasets](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/data)
* [Metapy Configs](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/config)
* [Search Engine Queries Testing](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/data/queries)
