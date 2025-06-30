# Sentiment Based Product Recommendation System


## Overview

The e-commerce landscape has undergone a dramatic transformation, with online platforms now dominating the retail space. Traditional brick-and-mortar models have given way to digital storefronts, enabling businesses to connect directly with consumers. Industry leaders like Amazon and Flipkart have set the benchmark, offering vast product selections and seamless user experiences.

In this competitive environment, emerging players like Ebuss are carving out their space by catering to a wide range of consumer needs—from household essentials to electronics. However, to thrive and compete with established giants, innovation is key. Enhancing user experience through intelligent, personalized services is no longer optional—it's essential.

To that end, Ebuss is leveraging machine learning to build a sentiment-driven product recommendation system. As a Machine Learning Engineer at Ebuss, your mission is to design and deploy a model that refines product recommendations based on user sentiment extracted from reviews.


## Project Objectives

**Data Collection & Sentiment Analysis**

Collect user reviews and ratings to analyze sentiment using natural language processing techniques.

**Recommendation Engine Development**
Build a robust recommendation system that incorporates sentiment insights to improve relevance.

**Sentiment-Enhanced Personalization**
Integrate sentiment analysis into the recommendation logic to deliver more personalized suggestions.

**End-to-End Deployment**
Deploy the solution with a user-friendly interface, ensuring smooth interaction and accessibility.

By adopting a sentiment-aware approach, Ebuss aims to exceed customer expectations, enhance satisfaction, and foster long-term loyalty.


## Solution Overview

**GitHub Repository:** [SentimentBasedProductRecommendation](https://github.com/sidhantdas789/sentiment_based_product_recommendation_system/tree/main)

**Tech Stack**

Python 3.11

scikit-learn 1.6.1

XGBoost 2.0.3

NumPy 1.26.4

NLTK 3.8.1

Pandas 2.2.2

Flask 3.0.2


## Solution Approach

**Data Preparation**
The dataset and attribute descriptions are provided in the project repository. Initial steps include data cleaning, visualization, and NLP-based text preprocessing.

**Text Vectorization**
TF-IDF is used to convert combined review titles and texts into numerical vectors, capturing the importance of words across the corpus.

**Handling Class Imbalance**
SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the sentiment classes before model training.

**Model Training & Evaluation**
Multiple classification models—Logistic Regression, Naive Bayes, Decision Tree, Random Forest, and XGBoost—are trained to classify sentiment as positive (1) or negative (0).
Evaluation metrics include Accuracy, Precision, Recall, F1 Score, and AUC. XGBoost outperforms others and is selected as the final model.

**Collaborative Filtering Recommender**
Both User-User and Item-Item collaborative filtering methods are implemented. RMSE is used for evaluation.

**Sentiment-Based Product Ranking**
The top 20 products are shortlisted using the recommender system. Sentiment predictions are made for all reviews, and the top 5 products with the highest positive sentiment are selected.

Model Deployment
Trained models are serialized using pickle and served via a Flask API. The front-end is built using Flask with Bootstrap and Jinja templates for a clean, responsive UI.
