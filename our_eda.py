import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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

def class_count_metrics(dataset, target, ngram_range=(1,1)):
    class_0 = dataset[dataset[target] == 0]['text']
    class_1 = dataset[dataset[target] == 1]['text']

    vectorizer = CountVectorizer(ngram_range=ngram_range)
    class_0_vec = vectorizer.fit_transform(class_0)
    class_1_vec = vectorizer.transform(class_1)

    vocab = np.array(vectorizer.get_feature_names_out())

    class_0_counts = np.asarray(class_0_vec.sum(axis=0)).flatten()
    class_1_counts = np.asarray(class_1_vec.sum(axis=0)).flatten()

    count_diff = class_1_counts - class_0_counts

    metrics_df = pd.DataFrame({
        'ngram': vocab,
        'class_0_count': class_0_counts,
        'class_1_count': class_1_counts,
        'count_diff': count_diff
    }).sort_values(by='count_diff', ascending=False)

    return metrics_df





# combined_sentiment_df = pd.read_csv("data_sentiment_preprocessed.csv")
# metrics = class_count_metrics(combined_sentiment_df, 'sentiment_label', ngram_range=(2,2))
# print(metrics.head())
    