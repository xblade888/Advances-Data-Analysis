import numpy as np
import scipy.sparse
from sklearn.decomposition import LatentDirichletAllocation

# Laden der TF-IDF-Matrix aus dem vorherigen Schritt
reviews = 'C:/temp/studium/TFIDF_tinder_reviews.npz'
matrix = scipy.sparse.load_npz(reviews)

# Laden der Feature-Wörtera aus dem vorherigen Schritt (TF-IDF)
features = 'C:/temp/studium/TFIDF_features.txt'
with open(features, 'r') as f:
    terms = np.array([line.strip() for line in f])

# Laden des LatentDirichletAllocation Performers, entdecken von 15 Themen und anwenden von LDA
n_components = 15  
modell = LatentDirichletAllocation(n_components=n_components, random_state=42)
matrix = modell.fit_transform(matrix)

# Jedes der 15 Tehmen wird durch 15 Wörter repräsentiert
topics = []
for idx, topic in enumerate(modell.components_):
    terms_in_topic = terms[np.argsort(topic)][:-16:-1]  # pro Thema wieder 15 Wörter 
    topics.append(terms_in_topic)
    print(f"Topic {idx}: {', '.join(terms_in_topic)}")

# Speicherung der Themen in einer Textdatatei
file = 'C:/temp/studium/LDA_topics.txt'
with open(file, 'w') as f:
    for idx, topic in enumerate(topics):
        f.write(f"Topic {idx}: {', '.join(topic)}\n")

# Speichert die LSA-Datei in einer Numpy-Datei + Erfolgsmeldung
lda_file = 'C:/temp/studium/LDA_themen.npy'
np.save(lda_file, modell.components_)
print("Topic-Modeling mit LDA abgeschlossen. In der neuen Datei LDA_themen.npy gespeichert")
