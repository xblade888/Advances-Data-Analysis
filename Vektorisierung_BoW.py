import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
import numpy as np

bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(bewertungen['review_text'])

bow_file = 'C:/temp/studium/BoW_tinder_reviews.npz'
scipy.sparse.save_npz(bow_file, bow_matrix)

print("Vektorisierierung mit Bag-of-Words abgeschlossen. In der neuen Datei BoW_tinder_reviews.npz gespeichert")
