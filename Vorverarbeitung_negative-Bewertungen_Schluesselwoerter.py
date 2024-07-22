import pandas as pd

# Datensatz laden (als CSV heruntergeladen)
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

# Filtern der Bewertungen nach 1+2 Sterne
bewertungen = bewertungen[bewertungen['review_rating'].isin([1, 2])]

# Schlüsselwörter nach denen gefiltert wird (Onlinerecherche)
schluesselwoerter = ['slow', 'lag', 'crash', 'freeze', 'bug', 'error', 'glitch', 'problem', 'fail', 'broken', 'frustrating', 'disappointing', 'poor', 'unusable',     'not working', 'stupid', 'bad', 'not loading', 'not working', 'connection issues', 'can not access', 'confusing', 'hard to use', 'difficult']

# Regex nach verschiedene Schlüsselwörter (ODER)
pattern = '|'.join(schluesselwoerter)
# Filtern nach Schlüsselwörtern, Groß- und Keinschreibung ergal
bewertungen = bewertungen[bewertungen['review_text'].str.contains(pattern, case=False, na=False)]

# Überschreiben der vorhandenen Datein und Erfolgsnachricht
bewertungen.to_csv('C:/temp/studium/TINDER_REVIEWS.csv', index=False)
print("Bewertungen nach Schluesselwoerter gefiltert, Datei überschrieben")
