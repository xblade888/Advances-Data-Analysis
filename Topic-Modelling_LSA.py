import numpy as np
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
import pandas as pd

BoW_reviews = 'C:/temp/studium/BoW_tinder_reviews.npz'
matrix = scipy.sparse.load_npz(BoW_reviews)

BoW_features = 'C:/temp/studium/BoW_features.txt'
with open(BoW_features, 'r') as f:
    terms = np.array([line.strip() for line in f])

# 15 Themen entdecken
n_components = 15 
modell = TruncatedSVD(n_components=n_components, random_state=42)
matrix = modell.fit_transform(matrix)

print("LSA Themen:")
for i, comp in enumerate(modell.components_):
    terms_in_comp = terms[np.argsort(comp)][:-16:-1]  # pro Thema 15 Wörter 
    print(f"Topic {i}: {', '.join(terms_in_comp)}")

lsa = 'C:/temp/studium/LSA_themen.npy'
np.save(lsa, matrix)
print("Tpic-Modelelling mit LSA abgeschlossen. In der neuen Datei LSA_themen.npy gespeichert")
