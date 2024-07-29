import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import numpy as np

# Reduzierung auf 1-2 Sterne Bewertungen und bekannte negative Schlüsselwörter
print("""Bewertungen werden nach neg. Schlüsselwörter und 1-2 Sterne Bewertungen reduziert
      
      """)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
bewertungen = bewertungen[bewertungen['review_rating'].isin([1, 2])]
schluesselwoerter = ['slow', 'lag', 'crash', 'freeze', 'bug', 'error', 'glitch', 'problem', 'fail', 'broken', 'frustrating', 'disappointing', 'poor', 'unusable', 'not working', 'stupid', 'bad', 'not loading', 'not working', 'connection issues', 'can not access', 'confusing', 'hard to use', 'difficult']
pattern = '|'.join(schluesselwoerter)
bewertungen = bewertungen[bewertungen['review_text'].str.contains(pattern, case=False, na=False)]
bewertungen.to_csv('C:/temp/studium/TINDER_REVIEWS.csv', index=False)
print("""Bewertungen werden nach neg. Schlüsselwörter und 1-2 Sterne Bewertungen reduziert -> Fortfahren mit der Weiterverarbeitung
      
      """)

# Weiterverarbeitung mit der Pandas Bibliothek und dem Python-Modul „re“
print("Bewertungen werden weiter berenigt")
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
def bereinigen(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\süäöÜÄÖ]', '', text)
    text = re.sub(r'http\S+', '', text)
    return text
bewertungen['review_text'] = bewertungen['review_text'].apply(bereinigen)
bewertungen.dropna(subset=['review_text'], inplace=True)
bewertungen.drop_duplicates(subset=['review_text'], inplace=True)
bewertungen.to_csv('C:/temp/studium/TINDER_REVIEWS.csv', index=False)
print("""Bewertungen erfolgreich gefiltert, Datei überschrieben
      
      """)

# Lemmatisierung und Entfernen von Stoppwörtern mit NLTK
print("Bewertungen werden nach lemmatisiert und Stoppwörter entfernt")
lemmatisierung = WordNetLemmatizer()
stoppwoerter = set(stopwords.words('english'))
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
def lemmatisierung_und_stoppwoerter_entfernen(text):
    woerter = text.split()
    bereinigte_woerter = [lemmatisierung.lemmatize(wort) for wort in woerter if wort not in stoppwoerter]
    return ' '.join(bereinigte_woerter)
bewertungen['review_text'] = bewertungen['review_text'].apply(lemmatisierung_und_stoppwoerter_entfernen)
bewertungen.dropna(subset=['review_text'], inplace=True)
bewertungen.drop_duplicates(subset=['review_text'], inplace=True)
bewertungen.to_csv('C:/temp/studium/TINDER_REVIEWS.csv', index=False)
print("""Bewertungen wurden Lemmatisiert und gefiltert -> Fortfahren mit Bag-of-Words (BoW) Vektorisierung
      
      """)

# Bag-of-Words (BoW)
print("Beginne mit der Vektorisierung mit BoW")
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(bewertungen['review_text'])
bow_file = 'C:/temp/studium/BoW_tinder_reviews.npz'
scipy.sparse.save_npz(bow_file, bow_matrix)
features_file = 'C:/temp/studium/BoW_features.txt'
with open(features_file, 'w') as f:
    for feature in vectorizer.get_feature_names_out():
        f.write(f"{feature}\n")
print("""BoW abgeschlossen. Informationen wurden in der BoW_tinder_reviews.npz gespeichert -> Fortfahren mit dem Topic Modelling mit LSA
      
      """)

# LSA
print("Anwenden von LSA")
reviews = 'C:/temp/studium/BoW_tinder_reviews.npz'
matrix = scipy.sparse.load_npz(reviews)
features = 'C:/temp/studium/BoW_features.txt'
with open(features, 'r') as f:
    terms = np.array([line.strip() for line in f])
n_components = 15 
modell = TruncatedSVD(n_components=n_components, random_state=42)
lsa = modell.fit_transform(matrix)
topics = []
for i, comp in enumerate(modell.components_):
    terms_in_comp = terms[np.argsort(comp)][:-16:-1] # pro Thema wieder 15 Wötert
    topics.append(terms_in_comp)
    print(f"Topic {i}: {', '.join(terms_in_comp)}")
file = 'C:/temp/studium/LSA_topics.txt'
with open(file, 'w') as f:
    for idx, topic in enumerate(topics):
        f.write(f"Topic {idx}: {', '.join(topic)}\n")
lsa_file = 'C:/temp/studium/LSA_themen.npy'
np.save(lsa_file, modell.components_)
print("""LSA abgeschlossen. Informationen wurden in der LSA_themen.npy gespeichert und LSA_topics.txt gespeichert -> Fortfahren mit dem Coherence Score
      
      """)

# Coherence Score LSA
print("""Berechne Coherence Score für LSA...
      
      """)
matrix = np.load(lsa_file)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
vectorizer = CountVectorizer()
term = vectorizer.fit_transform(bewertungen['review_text'])
with open(features_file, 'r') as f:
    terms = [line.strip() for line in f]
topics = []
for topic_idx in range(matrix.shape[0]):
    topic_terms = [terms[i] for i in np.argsort(matrix[topic_idx])[::-1][:15] if i < len(terms)]
    topics.append(topic_terms)
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
score = coherence(topics, vectorizer, term)
print(f'Coherence Score für LSA: {score}')
