import pandas as pd
import re

bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')

def bereinigung(bewertungen):
  bewertungen = bewertungen.lower()
  bewertungen = re.sub(r'[A-Za-z0-9üäöÜÄÖ], '', bewertungen)
  bewertungen = re.sub(r'http\S+', '', stringliteral)
  return bewertungen
