import json
import glob
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from GraphSum.src.utils.logging import init_logger, logger


def to_onehot(data, min_len):
    return np.bincount(data, minlength=min_len)


def load_dataset(data_path, shuffle):

    def _lazy_dataset_loader(pt_file):
        print('loading file %s' % pt_file)
        dataset = json.load(open(pt_file))
        return dataset

    pts = sorted(glob.glob(data_path + '/train/*.[0-9]*.json'))
    if pts:
        if shuffle:
            np.random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt)
    else:
        pt = sorted(glob.glob(data_path + '/train/*.json'))
        yield _lazy_dataset_loader(pt)


def train_sklearn_lda(data_path):
    dataset = load_dataset(data_path, False)

    lda = LatentDirichletAllocation(n_components=20, verbose=1)

    train_set = []
    for batch in dataset:
        for data in batch:
            train_set.append(data['tgt_str'])

    vectorizer = CountVectorizer(max_df=0.6, min_df=0.05)
    train_set = vectorizer.fit_transform(train_set)
    vocab = vectorizer.get_feature_names()

    lda.fit(train_set)

    topic_word = lda.components_
    n_topic_word = 10

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_topic_word + 1):-1]
        logger.info('Topic %d : %s' % (i, ' '.join(topic_words)))


def main():
    data_path = '../../../data/MultiNews'
    logger_file = '../../log/lda.log'
    init_logger(logger_file)

    train_sklearn_lda(data_path)


if __name__ == '__main__':
    main()
