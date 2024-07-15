import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Einlesen der LDA-Matrix und der Bewertungen
lda_file = 'C:/temp/studium/LDA_themen.npy'
matrix = np.load(lda_file)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

# Erstellen des CountVectorizers und Anpassen der Daten
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(bewertungen['review_text'])

# Einlesen der Feature-Namen
features_file = 'C:/temp/studium/TFIDF_features.txt'
with open(features_file, 'r') as f:
    terms = [line.strip() for line in f]

# Extrahieren der LDA-Themen
lda_topics = []
for topic_idx in range(matrix.shape[0]):
    topic_terms = [terms[i] for i in np.argsort(matrix[topic_idx])[::-1][:15] if i < len(terms)]
    lda_topics.append(topic_terms)

# Speichern der LDA-Themen in einer Datei
lda_topics_file = 'C:/temp/studium/LDA_topics.txt'
with open(lda_topics_file, 'w') as f:
    for idx, topic in enumerate(lda_topics):
        f.write(f"Topic {idx}: {', '.join(topic)}\n")

print(f"LDA-Themen wurden in der Datei {lda_topics_file} gespeichert.")

# Ausgabe der LDA-Themen in der Kommandozeile
print("LDA Themen:")
for idx, topic in enumerate(lda_topics):
    print(f"Topic {idx}: {', '.join(topic)}")

# Funktion zur Berechnung des Coherence Scores
def calculate_coherence(topics, vectorizer, X):
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

# Berechnung des Coherence Scores
lda_coherence_score = calculate_coherence(lda_topics, vectorizer, X)
print(f'Coherence Score für LDA: {lda_coherence_score}')
