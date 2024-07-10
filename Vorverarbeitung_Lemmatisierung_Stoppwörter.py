from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
lemmatisierung = WordNetLemmatizer()
stoppwoerter = set(stopwords.words('english'))

def lemmatisierung_und_stoppwörter_entfernen(text):
    woerter = text.split()
    bereinigte_woerter = [lemmatisierer.lemmatize(wort) for wort in woerter if wort not in stoppwoerter]
    return ' '.join(bereinigte_woerter)

bewertungen['review_text'] = bewertungen['review_text'].apply(lemmatisierung_und_stoppwörter_entfernen)


