import pandas as pd
import re

# Laden des Datensatzes aus dem vorherigen Schritt
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

# Funktion zur Bereinigung (Kleinbuchstaben, nicht-alphabetische Zeichen und URLs entfernen) und anschließende Rückgabe an das Hauptprogramm
def bereinigen(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\süäöÜÄÖ]', '', text)
    text = re.sub(r'http\S+', '', text)
    return text
# Anwenden der Bereinigungsfunktion auf Spalte "review_text"
bewertungen['review_text'] = bewertungen['review_text'].apply(bereinigen)

# Entfernt fehlende Werte + Duplikate
bewertungen.dropna(subset=['review_text'], inplace=True)
bewertungen.drop_duplicates(subset=['review_text'], inplace=True)

# Überschreiben der vorhandenen Datein und Erfolgsnachricht
bewertungen.to_csv('C:/temp/studium/TINDER_REVIEWS.csv', index=False)
print("Bewertungen erfolgreich gefiltert, Datei überschrieben")
