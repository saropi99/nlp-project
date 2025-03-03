from enum import Enum
import pandas as pd
import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords')
Stemmer = Enum('Stemmer', 'Porter WordNet')


def text_preprocess(text, remove_stopwords=True, stemmer=Stemmer.Porter):

    '''
    Steps:
        1. using lowercase
        2. deleting digits
        3. deleting double spaces
        4. removing stopwords (optional: True/False)
        5. deleting puntcuations
        6. stemming/lemmatization deploying function stem_text() -> returns result
    '''

    text = text.lower() # convert to lowercase
    text = re.sub(r'\d+', '', text)  # delete all digits
    text = re.sub(r'\s+', ' ', text)  # delete double spaces

    # stemming with stopwords removal
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    text = re.sub(r'[^\w\s]', '', text)  # remove punctuations but after removing stopwords bc of the stopwords list

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