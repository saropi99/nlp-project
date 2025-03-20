import re
import spacy
import contractions
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from enum import Enum

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

stop_words = set(stopwords.words('english'))
important_words = {"not", "no", "nor", "cannot", "again"}
stop_words -= important_words

nlp = spacy.load('en_core_web_sm')

Stemmer = Enum('Stemmer', 'Porter WordNet')

# Text Preprocessing Func.
def text_preprocess(text, remove_digits=True, stemmer=Stemmer.Porter):

    '''
    Steps:
        1. use just lowercases
        2. remove urls, emails from text
        3. remove digits (optional bc of the POS tagging)
        4. correct misspellings
        5. split contractions: can't into cannot, i'm into i am
        6. lemmatize with pos or stem, and remove stopwords except negations
        7. remove stopwords except negations (optional: True/False)
        8. delete punctuations
        9. delete double spaces, possible spaces at the beg. and end
    '''

    text = text.lower() # just lowercase
    text = re.sub(r'\S*https?:\S*', '', text) # remove possible urls
    text = re.sub(r'\S+@\S+', '', text) # remove possible emails

    if remove_digits:
        text = re.sub(r'\d+', '', text) # remove digits

    corrected_text = TextBlob(text).correct() # correct any misspellings
    text = str(corrected_text)
    text = contractions.fix(text) # split contractions

    if stemmer == Stemmer.WordNet:
        doc = nlp(text) # lemmatization + stopwords removing
        stemm_text = ' '.join([token.lemma_ for token in doc if token.text not in stop_words])

    if stemmer == Stemmer.Porter: # stemming + stopwords removing
        stemm_text = stem_text(text)

    stemm_text = re.sub(r'[^\w\s]', '', stemm_text) # delete punctuations (after stopwords removing just in case)
    stemm_text = re.sub(r'\s+', ' ', stemm_text).strip() # delete possible double spaces, at the beginning and end

    return stemm_text

# Stem Function
def stem_text(text):
    words = word_tokenize(text)
    clean = []
    s = PorterStemmer()
    for word in words:
        if word not in stop_words:
            stem = s.stem(word)
            clean.append(stem)
    return ' '.join(clean)