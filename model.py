from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.data.path.append('./nltk_data')
import zipfile
import os
import requests
from io import BytesIO

import nltk
import os

def download_nltk_data_from_file(file_path='nltk.txt'):
    if not os.path.exists(file_path):
        print(f"'{file_path}' not found. Skipping NLTK data download.")
        return

    with open(file_path, 'r') as f:
        resources = [line.strip() for line in f if line.strip()]

    for resource in resources:
        try:
            # Determine the correct path based on resource type
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading missing NLTK resource: {resource}")
            nltk.download(resource)

# Call the function at the start of your app
download_nltk_data_from_file()

class SentimentRecommenderModel:
    MODEL_URL = "https://raw.githubusercontent.com/sidhantdas789/sentiment_based_product_recommendation_system/main/pickle_files/sentiment-classification-xg-boost-model.pkl"
    VECTORIZER_URL = "https://raw.githubusercontent.com/sidhantdas789/sentiment_based_product_recommendation_system/main/pickle_files/tfidf_vectorizer.pkl"
    CLEANED_DATA_URL = "https://raw.githubusercontent.com/sidhantdas789/sentiment_based_product_recommendation_system/main/pickle_files/cleaned-data.pkl"
    USER_MATRIX_URL = "https://github.com/sidhantdas789/sentiment_based_product_recommendation_system/raw/main/pickle_files/user_final_rating.zip"

    def __init__(self):
        self._model = None
        self._vectorizer = None
        self._user_final_rating = None
        self._df = None

    @property
    def model(self):
        if self._model is None:
            r = requests.get(self.MODEL_URL)
            self._model = pickle.load(BytesIO(r.content))
        return self._model

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            r = requests.get(self.VECTORIZER_URL)
            self._vectorizer = pickle.load(BytesIO(r.content))
        return self._vectorizer

    @property
    def df(self):
        if self._df is None:
            r = requests.get(self.CLEANED_DATA_URL)
            self._df = pickle.load(BytesIO(r.content))
        return self._df

    @property
    def user_final_rating(self):
        if self._user_final_rating is None:
            r = requests.get(self.USER_MATRIX_URL)
            with zipfile.ZipFile(BytesIO(r.content)) as zip_ref:
                pkl_filename = zip_ref.namelist()[0]
                with zip_ref.open(pkl_filename) as f:
                    self._user_final_rating = pickle.load(f)
        return self._user_final_rating

    def get_sentiment_recommendations(self, user):
        if user not in self.user_final_rating.index:
            print(f"User name {user} doesn't exist")
            return None

        recommendations = list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
        temp = self.df[self.df.id.isin(recommendations)].copy()
        X = self.vectorizer.transform(temp["lemmatized_text"].values.astype(str))
        temp["predicted_sentiment"] = self.model.predict(X)
        temp = temp[["name", "predicted_sentiment"]]
        temp_grouped = temp.groupby("name", as_index=False).count()
        temp_grouped["pos_review_count"] = temp_grouped["name"].apply(
            lambda x: temp[(temp.name == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count()
        )
        temp_grouped["total_review_count"] = temp_grouped["predicted_sentiment"]
        temp_grouped["pos_sentiment_percent"] = (
            temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100
        ).round(2)
        return temp_grouped.sort_values("pos_sentiment_percent", ascending=False)[0:5]
