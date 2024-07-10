import pandas as pd
import re

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
print("Bewertungen erfolgreich gefiltert, Datei überschrieben")
