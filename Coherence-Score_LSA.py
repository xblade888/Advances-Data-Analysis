import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lsa_file = 'C:/temp/studium/LSA_themen.npy'
matrix = np.load(lsa_file)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(bewertungen['review_text'])

features = 'C:/temp/studium/BoW_features.txt'
with open(features, 'r') as f:
    terms = [line.strip() for line in f]

topics = []
for topic_idx in range(matrix.shape[0]):
    topic_terms = [terms[i] for i in np.argsort(matrix[topic_idx])[::-1][:15] if i < len(terms)]
    topics.append(topic_terms)

file = 'C:/temp/studium/LSA_topics.txt'
with open(file, 'w') as f:
    for idx, topic in enumerate(topics):
        f.write(f"Topic {idx}: {', '.join(topic)}\n")

print("LSA Themen:")
for idx, topic in enumerate(topics):
    print(f"Topic {idx}: {', '.join(topic)}")

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

lsa_coherence_score = calculate_coherence(topics, vectorizer, X)
print(f'Coherence Score für LSA: {lsa_coherence_score}')
