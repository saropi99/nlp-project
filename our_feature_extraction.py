from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def basic_bag(X_train, X_val, ohe=False, min_refs=None, ngram_range=(1, 1),  debug=False):
    vectorizer = CountVectorizer(ngram_range=ngram_range, binary=ohe)
    X_train_vec = vectorizer.fit_transform(X_train)
    if debug:
        print('Shape (X_train_vec) before reduction: ', X_train_vec.shape)
    
    word_counts = np.asarray(X_train_vec.sum(axis=0)).flatten()
    vocab = np.array(vectorizer.get_feature_names_out())
    selected_words = []
    if min_refs:
        selected_words = vocab[word_counts >= min_refs]
        vectorizer = CountVectorizer(vocabulary=selected_words)
        # Re transform training
        X_train_vec = vectorizer.fit_transform(X_train)
        if debug:
            print('Shape (X_train_vec) after reuction: ', X_train_vec.shape)
    X_val_vec = vectorizer.transform(X_val)
    if debug:
        print('Shape (X_val_vec): ', X_val_vec.shape)
    return word_counts, vocab, selected_words, vectorizer, X_train_vec, X_val_vec

def tf_idf(X_train, X_val, ohe=False, min_refs=None, ngram_range=(1, 1), debug=False):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, binary=ohe)
    X_train_vec = vectorizer.fit_transform(X_train)
    if debug:
        print('Shape (X_train_vec) before reduction: ', X_train_vec.shape)
    
    word_counts = np.asarray(X_train_vec.sum(axis=0)).flatten()
    vocab = np.array(vectorizer.get_feature_names_out())
    selected_words = []
    if min_refs:
        selected_words = vocab[word_counts >= min_refs]
        vectorizer = CountVectorizer(vocabulary=selected_words)
        # Re transform training
        X_train_vec = vectorizer.fit_transform(X_train)
        if debug:
            print('Shape (X_train_vec) after reuction: ', X_train_vec.shape)
    X_val_vec = vectorizer.transform(X_val)
    if debug:
        print('Shape (X_val_vec): ', X_val_vec.shape)
    return word_counts, vocab, selected_words, vectorizer, X_train_vec, X_val_vec
