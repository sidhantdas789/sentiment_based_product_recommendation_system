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
import zipfile
import os

def safe_nltk_download(resource_path):
    """
    Checks if the NLTK resource exists, and downloads it only if missing.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_path.split('/')[-1])

# Safely download required NLTK resources
safe_nltk_download('tokenizers/punkt')
safe_nltk_download('corpora/stopwords')
safe_nltk_download('taggers/averaged_perceptron_tagger')
safe_nltk_download('corpora/wordnet')
safe_nltk_download('corpora/omw-1.4')

class SentimentRecommenderModel:
    ROOT_PATH = "pickle_files/"
    MODEL = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf_vectorizer.pkl"
    USER_MATRIX = "user_final_rating.zip"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self._model = None
        self._vectorizer = None
        self._user_final_rating = None
        self._df = None

    @property
    def model(self):
        if self._model is None:
            with open(os.path.join(self.ROOT_PATH, self.MODEL), "rb") as f:
                self._model = pickle.load(f)
        return self._model

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            with open(os.path.join(self.ROOT_PATH, self.VECTORIZER), "rb") as f:
                self._vectorizer = pickle.load(f)
        return self._vectorizer

    @property
    def user_final_rating(self):
        if self._user_final_rating is None:
            zip_path = os.path.join(self.ROOT_PATH, self.USER_MATRIX)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                pkl_filename = zip_ref.namelist()[0]
                with zip_ref.open(pkl_filename) as f:
                    self._user_final_rating = pickle.load(f)
        return self._user_final_rating
   
    @property
    def df(self):
        if self._df is None:
            with open(os.path.join(self.ROOT_PATH, self.CLEANED_DATA), "rb") as f:
                self._df = pickle.load(f)
        return self._df

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
