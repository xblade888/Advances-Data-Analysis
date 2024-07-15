import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.sparse

bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(bewertungen['review_text'])

tfidf_file = 'C:/temp/studium/TFIDF_tinder_reviews.npz'
scipy.sparse.save_npz(tfidf_file, tfidf_matrix)

features_file = 'C:/temp/studium/TFIDF_features.txt'
with open(features_file, 'w') as f:
    for feature in vectorizer.get_feature_names_out():
        f.write(f"{feature}\n")

print("Vektorisierierung mit TF-IDF abgeschlossen. In der neuen Datei TFIDF_tinder_reviews.npz gespeichert")

