import numpy as np
import scipy.sparse
from sklearn.decomposition import LatentDirichletAllocation

tfidf_reviews = 'C:/temp/studium/TFIDF_tinder_reviews.npz'
matrix = scipy.sparse.load_npz(tfidf_reviews)

features = 'C:/temp/studium/TFIDF_features.txt'
with open(features, 'r') as f:
    terms = np.array([line.strip() for line in f])

# Analog zu BoW werden 15 Themen entdeckt
n_components = 15  
modell = LatentDirichletAllocation(n_components=n_components, random_state=42)
matrix = modell.fit_transform(matrix)

topics = []
for idx, topic in enumerate(modell.components_):
    terms_in_topic = terms[np.argsort(topic)][:-16:-1]  # pro Thema wieder 15 Wörter 
    topics.append(terms_in_topic)
    print(f"Topic {idx}: {', '.join(terms_in_topic)}")

file = 'C:/temp/studium/LDA_topics.txt'
with open(lda_topics_file, 'w') as f:
    for idx, topic in enumerate(topics):
        f.write(f"Topic {idx}: {', '.join(topic)}\n")

lda_file = 'C:/temp/studium/LDA_themen.npy'
np.save(lda_file, modell.components_)
print("Tpic-Modeling mit LDA abgeschlossen. In der neuen Datei LDA_themen.npy gespeichert")
