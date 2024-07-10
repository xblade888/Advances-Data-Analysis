import pandas as pd
bewertungen = pd.read_csv('C:/temp/studium/TINDER_REViEWS.csv')

bewertungen = bewertungen[bewertungen['review_rating'].isin(1,2)]

schluesselwoerter = ['slow', 'lag', 'crash', 'freeze', 'bug', 'error', 'glitch', 'problem', 'fail', 'broken', 'frustrating', 'disappointing', 'poor', 'unusable',     'not working', 'stupid', 'bad', 'not loading', 'not working', 'connection issues', 'can not access', 'confusing', 'hard to use', 'difficult']

pattern = '|'.join(schluesselwoerter)
bewertungen = bewertungen[bewertungen['review_text'].str.containts(pattern, case=False, na=False)]

bewertungen.to_csv('C:/temp/studium/TINDER_REViEWS.csv', index=False)
print("Neue Datei gespeichert")
