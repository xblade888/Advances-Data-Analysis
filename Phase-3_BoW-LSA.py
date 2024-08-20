import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pandarallel import pandarallel

print("Starten der Verarbeitung...")

# Funktion zum Datensatz laden und negative Bewertungen + Schluesselwoerter
def daten_laden_und_filtern(dateipfad):
    daten = pd.read_csv(dateipfad)
    daten = daten[daten['review_rating'].isin([1, 2])]
    schluesselwoerter = ['slow', 'lag', 'crash', 'freeze', 'bug', 'error', 'glitch', 'problem', 'fail', 'broken', 'frustrating', 'disappointing', 'poor', 'unusable',     'not working', 'stupid', 'bad', 'not loading', 'not working', 'connection issues', 'can not access', 'confusing', 'hard to use', 'difficult']
    muster = '|'.join(schluesselwoerter)
    daten = daten[daten['review_text'].str.contains(muster, case=False, na=False)]
    return daten

# Funktion zur Textbereinigung und Lemmatisierung (+ Pandarallel für die Pandas Bibliothek)
def texte_bereinigen_und_lemmatisieren(daten):
    pandarallel.initialize(progress_bar=True)
    def text_bereinigen(text):
        import re
        text = text.lower()
        text = re.sub(r'[^A-Za-z\süäöÜÄÖ]', '', text)
        text = re.sub(r'http\S+', '', text)
        return text

    lemmatisierer = WordNetLemmatizer()
    stoppwoerter = set(stopwords.words('english'))

    def lemmatisieren_und_stoppwoerter_entfernen(text):
        woerter = text.split()
        bereinigte_woerter = [lemmatisierer.lemmatize(wort) for wort in woerter if wort not in stoppwoerter]
        return ' '.join(bereinigte_woerter)

    # Parallele Ausführung
    daten['review_text'] = daten['review_text'].parallel_apply(text_bereinigen)
    daten['review_text'] = daten['review_text'].parallel_apply(lemmatisieren_und_stoppwoerter_entfernen)

    # leere Einträge und Duplikate entfernen
    daten.dropna(subset=['review_text'], inplace=True)
    daten.drop_duplicates(subset=['review_text'], inplace=True)

    return daten

# Funnktion für BoW und LSA
def bow_und_lsa(daten, bow_datei, feature_datei, lsa_datei, themen_datei):
    vektorisierer = CountVectorizer()
    bow_matrix = vektorisierer.fit_transform(daten['review_text'])

    scipy.sparse.save_npz(bow_datei, bow_matrix)

    with open(feature_datei, 'w') as f:
        for wort in vektorisierer.get_feature_names_out():
            f.write(f"{wort}\n")

    anzahl_themen = 15
    modell = TruncatedSVD(n_components=anzahl_themen, random_state=42)
    lsa_matrix = modell.fit_transform(bow_matrix)

    themen = []
    for i, komponente in enumerate(modell.components_):
        woerter_im_thema = vektorisierer.get_feature_names_out()[np.argsort(komponente)][:-16:-1]
        themen.append(woerter_im_thema)
        print(f"Thema {i}: {', '.join(woerter_im_thema)}")

    with open(themen_datei, 'w') as f:
        for idx, thema in enumerate(themen):
            f.write(f"Thema {idx}: {', '.join(thema)}\n")

    np.save(lsa_datei, modell.components_)

    return vektorisierer, bow_matrix, themen

# Coherence Score
def kohärenz_berechnen(vektorisierer, bow_matrix, themen):
    def kohärenz(themen, vektorisierer, matrix):
        kohärenz_scores = []
        for thema in themen:
            thema_indizes = [vektorisierer.vocabulary_.get(wort) for wort in thema if wort in vektorisierer.vocabulary_]
            if len(thema_indizes) < 2:
                continue
            thema_vektoren = matrix[:, thema_indizes].toarray()
            ähnlichkeit_matrix = cosine_similarity(thema_vektoren.T)
            kohärenz = np.mean(ähnlichkeit_matrix[np.triu_indices_from(ähnlichkeit_matrix, k=1)])
            kohärenz_scores.append(kohärenz)
        return np.mean(kohärenz_scores)

    # Berechnung und Ausgabe des Coherence Scores
    score = kohärenz(themen, vektorisierer, bow_matrix)
    print(f'Coherence Score für LSA: {score}')

    # Bewertung des Coherence Scores
    if score < 0.2:
        bewertung = "Sehr schlecht"
    elif 0.2 <= score < 0.4:
        bewertung = "Schlecht"
    elif 0.4 <= score < 0.6:
        bewertung = "Akzeptabel"
    elif 0.6 <= score < 0.8:
        bewertung = "Gut"
    else:
        bewertung = "Sehr gut"

    print(f'Die Bewertung des Coherence Scores ist: {bewertung}')

# Hauptprogramm, dass die Daten abspeichert und Funktionen ausführt
def main():
    dateipfad = 'C:/temp/studium/TINDER_REVIEWS.csv'
    bow_datei = 'C:/temp/studium/BoW_tinder_reviews.npz'
    feature_datei = 'C:/temp/studium/BoW_features.txt'
    lsa_datei = 'C:/temp/studium/LSA_themen.npy'
    themen_datei = 'C:/temp/studium/LSA_topics.txt'

    daten = daten_laden_und_filtern(dateipfad)
    daten = texte_bereinigen_und_lemmatisieren(daten)
    vektorisierer, bow_matrix, themen = bow_und_lsa(daten, bow_datei, feature_datei, lsa_datei, themen_datei)
    kohärenz_berechnen(vektorisierer, bow_matrix, themen)

if __name__ == "__main__":
    main()
