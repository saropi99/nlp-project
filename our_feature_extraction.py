from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing
from gensim.models.phrases import Phrases, Phraser
import torch
from transformers import BertTokenizer, BertModel

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def basic_bag(X_train, X_val, ohe=False, ngram_range=(1, 1), min_refs=None, debug=False):
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X_train_vec = vectorizer.fit_transform(X_train)
    if debug:
        print('Shape (X_train_vec): ', X_train_vec.shape)

    # Sum the word counts and extract vocabulary
    word_counts = np.asarray(X_train_vec.sum(axis=0)).flatten()
    vocab = np.array(vectorizer.get_feature_names_out())

    selected_words = []
    if ohe:
        if min_refs: # ohe=True, min_refs=True
            selected_words = vocab[word_counts >= min_refs]
            vectorizer = CountVectorizer(ngram_range=ngram_range, binary=ohe, vocabulary=selected_words)
        else: # ohe=True, min_refs=False
            vectorizer = CountVectorizer(ngram_range=ngram_range, binary=ohe)
        X_train_vec = vectorizer.fit_transform(X_train)
    elif min_refs: # ohe=False, min_refs=True
        selected_words = vocab[word_counts >= min_refs]
        vectorizer = CountVectorizer(ngram_range=ngram_range, vocabulary=selected_words)
        X_train_vec = vectorizer.fit_transform(X_train)
    if debug and len(selected_words) > 0:  # Only print after reduction if filtering happened
        print('Shape (X_train_vec) after reduction: ', X_train_vec.shape)
    X_val_vec = vectorizer.transform(X_val)
    if debug:
        print('Shape (X_val_vec): ', X_val_vec.shape)

    return word_counts, vocab, selected_words, vectorizer, X_train_vec, X_val_vec


def tf_idf(X_train, X_val, min_refs=None, ngram_range=(1, 1), debug=False):
    "TF-IDF Vectorizer"
    # Initialize CountVectorizer to count word/n-grams
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X_train_count = vectorizer.fit_transform(X_train)
    if debug:
        print('Shape (X_train_vec): ', X_train_count.shape)

    # Sum the word counts and extract vocabulary
    word_counts = np.asarray(X_train_count.sum(axis=0)).flatten()
    vocab = np.array(vectorizer.get_feature_names_out())

    # Filter vocabulary by minimum reference count (if provided)
    selected_words = []
    if min_refs:
        selected_words = vocab[word_counts >= min_refs]

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, vocabulary=selected_words if min_refs else None)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    if debug and len(selected_words):
        print('Shape (X_train_vec) after reduction: ', X_train_vec.shape)
    X_val_vec = tfidf_vectorizer.transform(X_val)
    if debug:
        print('Shape (X_val_vec): ', X_val_vec.shape)

    return word_counts, vocab, selected_words, tfidf_vectorizer, X_train_vec, X_val_vec

from collections import Counter
from typing import List, Dict, Tuple

def generate_ngrams(text: str, n: int) -> List[str]:
    """Generate n-grams from a given text."""
    tokens = text.split()  # Simple whitespace tokenizer
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

# Function for Word2vec fine tuning
def word2vec_alg(X_train, pretrained_path=None, vector_size=100, min_count=1, window=5, sg=False, epochs=30, alpha=0.025, alpha_min=0.0001, save_path=None):
    """
    Train a Word2Vec model on the given text, optionally using a pre-trained model.

    Parameters:
    X_train (list of lists of str): The training data, where each sentence is represented as a list of tokens (words).
    pretrained_path (str, optional): File path to a pre-trained Word2Vec model. If provided, the model will be fine-tuned.
    vector_size (int, default=100): The size (dimensionality) of the word vectors.
    min_count (int, default=1): Minimum frequency count for words to be included in the vocabulary.
    window (int, default=5): Maximum distance between the current and predicted word within a sentence.
    sg (bool, default=False): If True, use the Skip-gram model. If False, use the CBOW model.
    epochs (int, default=30): The number of training iterations (epochs).
    alpha (float, default=0.025): Initial learning rate for training.
    alpha_min (float, default=0.0001): Minimum learning rate after training.
    save_path (str, optional): If provided, the model will be saved at this path.

    Returns:

    model: The trained Word2Vec model.
    """

    X_train_list = list(X_train)
    phrases = Phrases(X_train_list, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sents = bigram[X_train_list]
    cores = multiprocessing.cpu_count() # check possible cores

    # the Word2Vec model with major parameters
    model = Word2Vec(vector_size=vector_size, min_count=min_count, window=window, sg=sg, workers=cores, alpha=alpha, min_alpha=alpha_min)

    model.build_vocab(sents, progress_per=10000) # vocabulary from the corpus
    total_examples = model.corpus_count  # number of examples in the corpus

    # use pretrained Word2Vec model
    if pretrained_path:
        pretrained_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=False) # upload pretrained model
        model.build_vocab([list(pretrained_model.key_to_index.keys())], update=True)  # add words from the pretrained model to the vocabulary
        model.wv.vectors = pretrained_model.vectors  # set the pretrained word vectors to the new model

        model.wv.locked = False  # False to fine-tune the pretrained vectors

    model.train(sents, total_examples=total_examples, epochs=epochs) # train the model

    if save_path:
        model.save(save_path)

    return model

def load_pretrained(path):
    pretrained_model = KeyedVectors.load_word2vec_format(path, binary=False) # upload pretrained model
    return pretrained_model

def glove_embedding(X_train, X_val, model, debug=False):
    def document_vector(doc):
        """Compute the mean word vector for a single document."""
        vectors = [model[word] for word in doc if word in model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    
    # Extract vocabulary and word counts
    vocab = list(model.key_to_index.keys())  # No need for `.wv`
    word_counts = [sum(1 for word in doc if word in model) for doc in X_train]
    selected_words = vocab  # All words in model vocabulary
    
    # Vectorize documents
    X_train_vec = np.array([document_vector(doc) for doc in X_train])
    X_val_vec = np.array([document_vector(doc) for doc in X_val])
    
    if debug:
        print("Shape (X_train_vec):", X_train_vec.shape)
        print("Shape (X_val_vec):", X_val_vec.shape)
    
    return word_counts, vocab, selected_words, document_vector, X_train_vec, X_val_vec

def bert_embeddings(text_list, model=bert_model, tokenizer=bert_tokenizer):
    """Tokenize text and generate averaged BERT embeddings."""
    model.eval()  # Set the model to evaluation mode
    embeddings = []
    
    with torch.no_grad():  # Disable gradient computation
        for text in text_list:
            encoded_input = tokenizer(text, padding=True, return_tensors='pt', truncation=True, max_length=512)
            output = model(**encoded_input)
            last_hidden_state = output.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
            avg_embedding = last_hidden_state.mean(dim=1)  # Average over the sequence length dimension
            embeddings.append(avg_embedding.squeeze().numpy())  # Convert to NumPy array

    return torch.tensor(embeddings)  # Return as tensor


def bert_embeddings_split(X_train, X_val):
    # Generate embeddings
    X_train_vec = bert_embeddings(X_train)
    X_val_vec = bert_embeddings(X_val)
    return X_train_vec, X_val_vec

class ClassAwareVectorizer:
    def __init__(self, n: int = 1, ohe: bool = False):
        """
        Class-Aware Vectorizer for Sentiment Analysis with Improved Scaling.

        Args:
            n (int): N-gram size.
            ohe (bool): If True, uses one-hot encoding weights; otherwise, multiplies by token frequency.
        """
        self.n = n
        self.ohe = ohe
        self.vocab = {}
        self.word_weights = {}
        self.mean_weight = 0
        self.max_abs_deviation = 1  # Default values

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
        
        # Compute class-aware raw weights
        full_vocab = set(class_0_counts.keys()).union(set(class_1_counts.keys()))
        raw_weights = {
            word: (class_1_counts[word] - class_0_counts[word]) / (class_1_counts[word] + class_0_counts[word] + 1e-6) 
            for word in full_vocab
        }
        
        # Compute mean and max absolute deviation
        self.mean_weight = np.mean(list(raw_weights.values()))
        self.max_abs_deviation = max(abs(w - self.mean_weight) for w in raw_weights.values()) + 1e-6
        
        # Normalize weights to range [0,1] centered at 0.5
        self.word_weights = {
            word: 0.5 + 0.5 * (weight - self.mean_weight) / self.max_abs_deviation
            for word, weight in raw_weights.items()
        }
        
        # Create vocabulary index
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
        vectors = np.full((len(docs), len(self.vocab)), 0.5)  # Default to neutral value

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
