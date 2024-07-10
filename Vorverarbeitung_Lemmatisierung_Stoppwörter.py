from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

bewertungen = pd.read_csv('C:/temp/studium/TINDER_REVIEWS.csv')
lemmatisierung = WordNetLemmatizer()
stoppwoerter = set(stopwords.words('english'))
