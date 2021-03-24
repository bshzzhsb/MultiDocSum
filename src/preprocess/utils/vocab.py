import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def build_count_vectorizer(dataset, stop_words):
    vectorizer = CountVectorizer(max_df=args.max_df, min_df=args.min_df, stop_words=stop_words)
    dataset = vectorizer.fit_transform(dataset)
    vocab = dataset.get_feature_names()

    return dataset, vocab


def get_nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy()
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    most_similar = []
    [most_similar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = most_similar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors
