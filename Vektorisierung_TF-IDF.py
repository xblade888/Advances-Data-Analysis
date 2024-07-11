import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(bewertungen['review_text'])

tfidf_file = 'C:/temp/studium/TFIDF_tinder_reviews.npz'

print("done")
