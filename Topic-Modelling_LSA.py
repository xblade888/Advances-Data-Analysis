import numpy as np
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
import pandas as pd

# Laden der BoW-Matrix aus dem vorherigen Schritt
reviews = 'C:/temp/studium/BoW_tinder_reviews.npz'
matrix = scipy.sparse.load_npz(reviews)

# Laden der Feature-Wörtera aus dem vorherigen Schritt (BoW)
features = 'C:/temp/studium/BoW_features.txt'
with open(features, 'r') as f:
    terms = np.array([line.strip() for line in f])

# Laden des TruncatedSVD Performers, entdecken von 15 Themen und anwenden von LSA
n_components = 15 
modell = TruncatedSVD(n_components=n_components, random_state=42)
lsa = modell.fit_transform(matrix)

# Jedes der 15 Tehmen wird durch 15 Wörter repräsentiert
topics = []
for i, comp in enumerate(modell.components_):
    terms_in_comp = terms[np.argsort(comp)][:-16:-1]  # pro Thema 15 Wörter 
    topics.append(terms_in_comp)
    print(f"Topic {i}: {', '.join(terms_in_comp)}")

# Speicherung der Themen in einer Textdatatei
file = 'C:/temp/studium/LSA_topics.txt'
with open(file, 'w') as f:
    for idx, topic in enumerate(topics):
        f.write(f"Topic {idx}: {', '.join(topic)}\n")

# Speichert die LSA-Datei in einer Numpy-Datei + Erfolgsmeldung
lsa_file = 'C:/temp/studium/LSA_themen.npy'
np.save(lsa_file, modell.components_)
print("Tpic-Modeling mit LSA abgeschlossen. In der neuen Datei LSA_themen.npy gespeichert")
