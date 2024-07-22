import spacy # Nutzung der Spacy Bibiliothek, da die Installation von Gensim nicht möglich ist
import pandas as pd

# Lade das spaCy Modells
nlp = spacy.load('en_core_web_md')

# Datensatz laden (als CSV heruntergeladen und Vorverarbeitet)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

# Funktion zur Verarbeitung der Texte mit dem Spacy-Modell
def vector(texts):
    return [nlp(text).vector for text in texts]

# Verarbeitung des Datensatzes und Speicherung
bewertungen['vector'] = vector(bewertungen['review_text'])

# Speichere die Vektoren in einer CSV-Datei
csv_path = "C:/temp/studium/spacy_vectors.csv"
bewertungen['vector'].apply(pd.Series).to_csv(csv_path, index=False)
print('Vektoren in C:/temp/studium/spacy_vectors.csv gespeichert')
