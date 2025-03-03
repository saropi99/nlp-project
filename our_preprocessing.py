from enum import Enum
import pandas as pd
import re
import textblob
import contractions

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords')
Stemmer = Enum('Stemmer', 'Porter WordNet')


def text_preprocess(text, remove_stopwords=True, remove_digits=True, stemmer=Stemmer.Porter):

    '''
    Steps:
        1. using lowercase
        2. removing urls, emails
        3. removing digits (optional bc of the POS tagging)
        4. correcting misspellings
        5. split contractions: can't into cannot, i'm into i am
        6. removing stopwords (optional: True/False)
        7. deleting puntcuations
        8. deleting double spaces
        9. stemming/lemmatization deploying function stem_text() -> returns result
    '''

    text = text.lower() # just lowercase
    text = re.sub(r'\S*https?:\S*', '', text) # remove urls
    text = re.sub(r'\S+@\S+', '', text) # remove emails

    if remove_digits:
        text = re.sub(r'\d+', '', text)  # remove digits

    corrected_text = textblob.TextBlob(text).correct() # correct misspellings
    text = contractions.fix(str(corrected_text)) # split contractions

    # stemming with stopwords removal
    if remove_stopwords:
        important_words = {"not", "no", "nor", "cannot"}
        stop_words = set(stopwords.words('english'))
        stop_words -= important_words
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    text = re.sub(r'[^\w\s]', '', text) # delete punctuations (after stopwords removing just in case)
    text = re.sub(r'\s+', ' ', text)  # delete double spaces

    return stem_text(text, stemmer)

def stem_text(text, stemmer=Stemmer.Porter):

    words = word_tokenize(text)
    clean = []
    s = WordNetLemmatizer() if stemmer == Stemmer.WordNet else PorterStemmer()
    for word in words:
        if stemmer == Stemmer.WordNet:
            stem = s.lemmatize(word)
        else:
            stem = s.stem(word)
        clean.append(stem)
    return clean

def class_distribution(dataset):
    if 'sentiment_label' in dataset.columns:
        counts = dataset.sentiment_label.value_counts()
    elif 'sarcasm_label' in dataset.columns:
        counts = dataset.sarcasm_label.value_counts()
    else:
        print('No relevant labels found in DataFrame.')
        return
    percentages = counts / counts.sum() * 100
    distribution_df = pd.DataFrame({'Count': counts, 'Percentage': percentages.round(2)})
    print(distribution_df)