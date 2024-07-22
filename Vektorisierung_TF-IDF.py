import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.sparse

# Datensatz laden (als CSV heruntergeladen und Vorverarbeitet)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

# Laden des Vecotirzers und Umwandlung der Spalte "review_text" in TF-IDF-Matrix 
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(bewertungen['review_text'])

# Speichert die TF-IDF-Matrix in einer Numpy Datei
tfidf_file = 'C:/temp/studium/TFIDF_tinder_reviews.npz'
scipy.sparse.save_npz(tfidf_file, tfidf_matrix)

# Speichert einzigartige Wörter in eine Textdatei
features_file = 'C:/temp/studium/TFIDF_features.txt'
with open(features_file, 'w') as f:
    for feature in vectorizer.get_feature_names_out():
        f.write(f"{feature}\n")

# Erfolgsmeldung das die Vektorisierung abgeschlossen wurde, damit kann das LDA-Script gestartet werden
print("Vektorisierierung mit TF-IDF abgeschlossen. In der neuen Datei TFIDF_tinder_reviews.npz gespeichert")
