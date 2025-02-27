from nltk import word_tokenize
from enum import Enum

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

Stemmer = Enum('Stemmer', 'Porter WordNet')

def tokenize_text(text, stemmer=Stemmer.Porter):
    if stemmer == Stemmer.WordNet:
        s = WordNetLemmatizer()
    else:
        s = PorterStemmer()
    words = word_tokenize(text)
    clean = []
    for word in words:
        if stemmer == Stemmer.WordNet:
            stem = s.lemmatize(word)
        else:
            stem = s.stem(word)
        clean.append(stem)
    return clean