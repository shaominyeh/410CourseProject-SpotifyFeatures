# Spotify Features App

## CS 410 Course Project Fall 2023: Utilizing text mining and machine learning to gain insights on Spotify songs

By: Shao-Min Yeh (snyeh2@illinois.edu)

## Deliverables 

This section lists all the required deliverables for the class project:

* [Project Proposal](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/blob/main/Project%20Proposal.pdf)
* [Progress Report](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/blob/main/Project%20Progress%20Report.pdf)
* [Code](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/src)
* Final Documentation WIP
* Presentation WIP

## Project Overview

This project consists of three different models regarding songs from a Spotify dataset all encapsulated by a Flask application. The TF-IDF Search Engine model takes a user-submitted query and outputs the most similar songs based on TF-IDF weighting. The Similar Songs model takes a user-submitted song and outputs the most similar songs based on TF-IDF and musical features. The Sentiment Analysis model takes a user-submitted song and outputs the sentiment analysis based on that song. 

## Project Installation
This section consists of the necessary installations to run this project:

### Repository Cloning
After creating a new folder in your coding environment, simply clone the git repository:
```
git clone https://github.com/shaominyeh/410CourseProject-SpotifyFeatures.git
```

### System Environment
Metapy requires Python 3.5.6 to run (in Windows), so we will utilize [Anaconda](https://www.anaconda.com/download) to make a virtual environment.  

First, open up an Anaconda Prompt. Then, create a new Conda Environment with Python 3.5:
```
conda create -n SpotifyApp
conda activate SpotifyApp
conda install python=3.5
```
Within the repository now, select the Python Interpreter and click on your python environment. In VS Code, press F1 -> type "Python: Select Interpreter" -> press Python 3.5.6 ('SpotifyApp')

### Library Requirements
If the above is done correctly, all you need to do is input the following for library requirements:
```
pip install -r requirements.txt
```
More specifically, in the Conda environment, it should look like the following:
```
(SpotifyApp) C:\Users\(user)>pip install -r requirements.txt
```

## Project Development
After the project installation is done, you should have a development environment containing the repository, Conda environment, and respective libraries. After this, you need to open a cmd terminal (if on Windows) and run the commands listed below. In that environment, the rest of the commands will need you to type:
```
cd src
``` 

### Flask Application
If everything is setup correctly, typing the following command and accessing http://127.0.0.1:5000/ should lead to the webpage.
```
python app.py
```
The webpage has their respective buttons and instructions regarding different options and when a user needs to input a query.

### Model Application
All three models can be accessed in the same way: changing the main block variables that are all caps. For example:
```python
# For example, change the following variables:
if __name__ == '__main__':
    CHOSEN_INDEX = 0
    UNIFIED_STATE = 47
    IS_TOKENIZED_QUERY = False
```
```
# Change the file to your respective model
python sentiment_analysis.py
```

## Modeling Details
This section consists of how models were implemented:

* Search Engine: This model utilizes Metapy for the TF-IDF ranking and Pandas to preprocess the dataset and catalog directories. This is done with a BM25 ranking model with Metapy's implementation. Along with this, catalog directories from the preprocessing allow for separated or combined rankings (for title/artist rankings).
* Similar Songs: This model utilizes Scikit-Learn functions (TF-IDF vectorizer and Nearest Neighbors) for ranking functions and Pandas for preprocessing. This is done with a KNN model with cosine-similarity. Along with this, TF-IDF is calculated with Scikit functions and musical features are calculated directly from the dataset.
* Sentiment Analysis: This model utilizes Scikit-Learn functions (Linear Regression, metrics, and Train-Test splits) for ranking functions, Metapy for query tokenization, and Pandas for preprocessing. This model trains with the dataset's valence (emotional value) with Linear Regression to train the model and calculate predictions. 

## Other Important Pages

Here are the rest of the important directory paths found in this repo:

* [HTML Templates](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/templates)
* [Sources](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/blob/main/docs/sources.txt)
* [CSV Datasets](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/data)
* [Metapy Configs](https://github.com/shaominyeh/410CourseProject-SpotifyFeatures/tree/main/config)
