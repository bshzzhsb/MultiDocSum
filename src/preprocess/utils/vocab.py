import argparse
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from data import load_stop_words, data_loader


def build_count_vectorizer(dataset, stop_words):
    vectorizer = CountVectorizer(max_df=args.max_df, min_df=args.min_df, stop_words=stop_words)
    dataset = vectorizer.fit_transform(dataset)
    vocab = dataset.get_feature_names()

    return dataset, vocab


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
