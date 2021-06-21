#Michał Olejek - Fake news detection program

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#Read the data
df=pd.read_csv('news.csv')
#geting the labels
labels=df.label
labels.head()

#Split data set for train and test set
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.3, random_state=42)
#creating TF-IDF 
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.78)

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)

print(f'Accuracy: {round(score*100,2)}%')