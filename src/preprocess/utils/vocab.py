import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from data import load_stop_words, data_loader


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


def main():
    stop_words = load_stop_words(args.stop_words_file)
    dataset = data_loader(args.data_path)

    build_count_vectorizer(dataset, stop_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_df', default=0.7, type=float)
    parser.add_argument('--min_df', default=100, type=int)
    parser.add_argument('--stop_words_file', default='../../../files/stop_words.txt', type=str)

    args = parser.parse_args()

    main()
