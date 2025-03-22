import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def visualize_word_embeddings(model, words_explore_list):
    my_word_list = []
    my_word_vectors = []
    # if one of words from the words_explore_list is not in the model vocab skip it
    for word in words_explore_list:
        try:
            my_word_vectors.append(model.wv[word])
            my_word_list.append(word)
        except KeyError:
            continue

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    result = pca.fit_transform(my_word_vectors)

    # Display 2d plot
    plt.figure(figsize=(10, 8))
    plt.scatter(result[:, 0], result[:, 1])

    # annotating the words on the plot
    for i, word in enumerate(my_word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=8)

    plt.title("Word Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()
