import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(bewertungen['review_text'])

bow_file = 'C:/temp/studium/BoW_cleaned_tinder_reviews.npz'
np.savez_compressed(sparse_matrix_file, data=bow_matrix.data, indices=bow_matrix.indices, indptr=bow_matrix.indptr, shape=bow_matrix.shape)
print(f"Vektorisierung mit Bag-of-Words abgeschlossen. Datei: {bow_file}")

print("feritg")
