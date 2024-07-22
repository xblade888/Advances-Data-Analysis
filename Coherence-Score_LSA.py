import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Laden des LSA-Modeslls  aus dem vorherigen Schritt + Laden den bereinigten Bewertungsdatei 
lsa = 'C:/temp/studium/LSA_themen.npy'
matrix = np.load(lsa)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

# Laden des Vectoriziers
vectorizer = CountVectorizer()
term = vectorizer.fit_transform(bewertungen['review_text'])

# Laden der Feature-Wörter aus dem vorherigen Schritt
features = 'C:/temp/studium/BoW_features.txt'
with open(features, 'r') as f:
    terms = [line.strip() for line in f]

# Speichern des Hautptwöórtes jedes Thema aus der LSA-Matrix
topics = []
for topic_idx in range(matrix.shape[0]):
    topic_terms = [terms[i] for i in np.argsort(matrix[topic_idx])[::-1][:15] if i < len(terms)]
    topics.append(topic_terms)

# Speichern in einer Textdatatei uns Ausgabe in der CLI
file = 'C:/temp/studium/LSA_topics.txt'
with open(file, 'w') as f:
    for idx, topic in enumerate(topics):
        f.write(f"Topic {idx}: {', '.join(topic)}\n")

print("LSA Themen:")
for idx, topic in enumerate(topics):
    print(f"Topic {idx}: {', '.join(topic)}")

# Funktion für den Coherence Score mit Rückgabe an das Hauptprogramm
def coherence(topics, vectorizer, X):
    coherence_scores = []
    for topic in topics:
        topic_indices = [vectorizer.vocabulary_.get(term) for term in topic if term in vectorizer.vocabulary_]
        if len(topic_indices) < 2:
            continue
        topic_vectors = X[:, topic_indices].toarray()
        sim_matrix = cosine_similarity(topic_vectors.T)
        coherence = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
        coherence_scores.append(coherence)
    return np.mean(coherence_scores)

# Berechnung des durschschnittlichen Scores uns Ausgabe in der CLI
score = coherence(topics, vectorizer, term)
print(f'Coherence Score für LSA: {score}')
