import json
import glob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def load_stop_words(stop_words_file):
    with open(stop_words_file, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file.readlines()]
    return stop_words


def data_loader(data_path, phase='*', source='tgt', spm=None):

    def dataset_loader(pt_file):
        print('loading file %s' % pt_file)
        dataset = json.load(open(pt_file))
        return dataset

    pts = sorted(glob.glob(data_path + '/' + phase + '/*.[0-9]*.json'))
    assert len(pts) > 0
    np.random.shuffle(pts)

    train_dataset = []
    if source == 'tgt':
        for pt in pts:
            data = dataset_loader(pt)
            for item in data:
                train_dataset.append(item['tgt_str'])
    elif source == 'src':
        for pt in pts:
            data = dataset_loader(pt)
            for item in data:
                for src in item['src']:
                    train_dataset.append(spm.DecodeIds(src))
    elif source == 'all':
        for pt in pts:
            data = dataset_loader(pt)
            for item in data:
                train_dataset.append(item['tgt_str'])
                for src in item['src']:
                    train_dataset.append(spm.DecodeIds(src))
    else:
        raise NotImplementedError('source must in ["tgt", "src"]')

    return train_dataset


def build_count_vectorizer(dataset, stop_words, max_df, min_df):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    dataset = vectorizer.fit_transform(dataset)
    vocab = vectorizer.get_feature_names()

    return dataset, vocab


def get_count_vectorizer(vocab):
    vectorizer = CountVectorizer(vocabulary=vocab)
    return vectorizer


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
