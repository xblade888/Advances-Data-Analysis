import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Laden des Datensatzes aus dem vorherigen Schritt
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
# Lemmatizer initalisieren mit Variable + englische Stoppwörter laden
lemmatisierung = WordNetLemmatizer()
stoppwoerter = set(stopwords.words('english'))

# Funktion zur Lemmatisierung und entfernen von Stoppwörtern
def lemmatisierung_und_stoppwörter_entfernen(text):
    woerter = text.split() # Liste von Wrntern
    bereinigte_woerter = [lemmatisierung.lemmatize(wort) for wort in woerter if wort not in stoppwoerter] # Zusammenfügen
    return ' '.join(bereinigte_woerter)

# Anwenden der Bereinigungsfunktion auf Spalte "review_text"
bewertungen['review_text'] = bewertungen['review_text'].apply(lemmatisierung_und_stoppwörter_entfernen)

# Entfernt fehlende Werte + Duplikate
bewertungen.dropna(subset=['review_text'], inplace=True)
bewertungen.drop_duplicates(subset=['review_text'], inplace=True)

# Überschreiben der vorhandenen Datein und Erfolgsnachricht
bewertungen.to_csv('C:/temp/studium/TINDER_REVIEWS.csv', index=False)
print("Bewertungen gefiltert, Datei überschrieben, damit ist die Vorverarbeitung abgeschlossen")
