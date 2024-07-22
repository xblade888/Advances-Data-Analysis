import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
import numpy as np

# Datensatz laden (als CSV heruntergeladen und Vorverarbeitet)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

# Laden des Vecotirzers und Umwandlung der Spalte "review_text" in BoW 
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(bewertungen['review_text'])

# Speichert die BoW-Matrix in einer Numpy Datei
bow_file = 'C:/temp/studium/BoW_tinder_reviews.npz'
scipy.sparse.save_npz(bow_file, bow_matrix)

# Speichert einzigartige Wörter in eine Textdatei
features_file = 'C:/temp/studium/BoW_features.txt'
with open(features_file, 'w') as f:
    for feature in vectorizer.get_feature_names_out():
        f.write(f"{feature}\n")

# Erfolgsmeldung das die Vektorisierung abgeschlossen wurde, damit kann das LSA-Script gestartet werden
print("Vektorisierierung mit Bag-of-Words abgeschlossen. In der neuen Datei BoW_tinder_reviews.npz gespeichert")
