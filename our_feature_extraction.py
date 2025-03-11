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

from collections import Counter
from typing import List, Dict, Tuple

def generate_ngrams(text: str, n: int) -> List[str]:
    """Generate n-grams from a given text."""
    tokens = text.split()  # Simple whitespace tokenizer
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

class ClassAwareVectorizer:
    def __init__(self, n: int = 1, ohe: bool = False):
        """
        Class-Aware Vectorizer for Sentiment Analysis.
        
        Args:
            n (int): N-gram size.
            ohe (bool): If True, uses one-hot encoding weights; otherwise, multiplies by token frequency.
        """
        self.n = n
        self.ohe = ohe
        self.vocab = {}
        self.word_weights = {}
        self.min_weight = 0
        self.max_weight = 1 
    
    def fit(self, docs: List[str], labels: List[int]):
        """
        Learns class-aware word weights from the training data.
        
        Args:
            docs (List[str]): Training documents.
            labels (List[int]): Corresponding class labels (0 or 1).
        """
        class_0_counts = Counter()
        class_1_counts = Counter()
        
        # Count n-grams per class
        for doc, label in zip(docs, labels):
            ngrams = generate_ngrams(doc, self.n)
            if label == 0:
                class_0_counts.update(ngrams)
            else:
                class_1_counts.update(ngrams)
        
        # Merge vocabularies and compute class-aware weights
        full_vocab = set(class_0_counts.keys()).union(set(class_1_counts.keys()))
        
        self.word_weights = {
            word: (class_1_counts[word] - class_0_counts[word]) / (class_1_counts[word] + class_0_counts[word] + 1e-6) 
            for word in full_vocab
        }
        
        # Learn min and max values for scaling
        self.min_weight = min(self.word_weights.values(), default=0)
        self.max_weight = max(self.word_weights.values(), default=1)
        
        # Normalize weights to [0, 1]
        self.word_weights = {
            word: 0.5 + 0.5 * (weight - self.min_weight) / (self.max_weight - self.min_weight + 1e-6)
            for word, weight in self.word_weights.items()
        }
        

        self.vocab = {word: i for i, word in enumerate(full_vocab)}
        
        return self  # Allows method chaining

    def transform(self, docs: List[str]) -> np.ndarray:
        """
        Transforms input texts into fixed-length vectors.
        
        Args:
            docs (List[str]): List of input documents.
        
        Returns:
            np.ndarray: Transformed feature matrix (num_samples x vocab_size).
        """
        vectors = np.zeros((len(docs), len(self.vocab)))

        for i, doc in enumerate(docs):
            ngrams = generate_ngrams(doc, self.n)
            doc_count = Counter(ngrams)

            for word, count in doc_count.items():
                if word in self.vocab:
                    weight = self.word_weights[word]
                    vectors[i, self.vocab[word]] = weight if self.ohe else weight * count
        
        return vectors

    def fit_transform(self, docs: List[str], labels: List[int]) -> np.ndarray:
        """
        Combines fit and transform.
        
        Args:
            docs (List[str]): Training documents.
            labels (List[int]): Corresponding class labels.
        
        Returns:
            np.ndarray: Transformed feature matrix.
        """
        return self.fit(docs, labels).transform(docs)

# # Example usage:
# docs = ["I love this movie", "This movie is terrible", "Amazing acting and plot", "Worst film ever"]
# labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# vectorizer = ClassAwareVectorizer(n=1, ohe=False)
# X_train = vectorizer.fit_transform(docs, labels)

# print("Vocabulary:", vectorizer.vocab)
# print("Feature Matrix:\n", X_train)

# # Test with new data
# new_texts = ["Amazing movie and plot", "Terrible terrible film"]
# X_test = vectorizer.transform(new_texts)
# print("New Texts Feature Matrix:\n", X_test)

