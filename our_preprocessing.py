from enum import Enum
import pandas as pd
import re
import textblob
import contractions

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer



nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


Stemmer = Enum('Stemmer', 'Porter WordNet')

def text_preprocess(text, remove_stopwords=True, remove_digits=True, stemmer=Stemmer.Porter):

    '''
    Steps:
        1. using lowercases
        2. removing urls, emails from text
        3. removing digits (optional bc of the POS tagging)
        4. correcting misspellings
        5. splitting contractions: can't into cannot, i'm into i am
        6. stemming / pos + lemmatization using func. stem_text()
        7. removing stopwords (optional: True/False)
        8. deleting puntcuations
        9. deleting double spaces, possible spaces at the beg. and end
    '''

    text = text.lower() # just lowercase
    text = re.sub(r'\S*https?:\S*', '', text) # remove urls
    text = re.sub(r'\S+@\S+', '', text) # remove emails

    if remove_digits:
        text = re.sub(r'\d+', '', text)  # remove digits

    corrected_text = textblob.TextBlob(text).correct() # correct misspellings
    text = contractions.fix(str(corrected_text)) # split contractions

    text = stem_text(text, stemmer) # stemmer/lemmatizer

    # stopwords removal
    if remove_stopwords:
        important_words = {"not", "no", "nor", "cannot"}
        stop_words = set(stopwords.words('english'))
        stop_words -= important_words
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    text = re.sub(r'[^\w\s]', '', text) # delete punctuations (after stopwords removing just in case)
    text = re.sub(r'\s+', ' ', text).strip()   # delete double spaces, at the beginning and end

    return text

def stem_text(text, stemmer=Stemmer.Porter):

    words = word_tokenize(text)
    clean = []

    if stemmer == Stemmer.WordNet:
        s = WordNetLemmatizer()
        pos_tags = pos_tag(words)
        for word, tag in pos_tags:
            stem = s.lemmatize(word, get_wordnet_pos(tag))
            clean.append(stem)
    else:
        s = PorterStemmer()
        for word in words:
            stem = s.stem(word)
            clean.append(stem)

    return ' '.join(clean)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ  # Adjective
    elif tag.startswith('V'):
        return wordnet.VERB  # Verb
    elif tag.startswith('N'):
        return wordnet.NOUN  # Noun
    elif tag.startswith('R'):
        return wordnet.ADV  # Adverb
    else:
        return wordnet.NOUN

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