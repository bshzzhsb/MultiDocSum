import json
import glob
import numpy as np


def load_stop_words(stop_words_file):
    with open(stop_words_file, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file.readlines()]
        file.close()
    return stop_words


def data_loader(data_path, phase='*'):

    def dataset_loader(pt_file):
        print('loading file %s' % pt_file)
        dataset = json.load(open(pt_file))
        return dataset

    pts = sorted(glob.glob(data_path + '/' + phase + '/*.[0-9]*.json'))
    assert len(pts) > 0
    np.random.shuffle(pts)

    train_dataset = []
    for pt in pts:
        data = dataset_loader(pt)
        for item in data:
            train_dataset.append(item['tgt_str'])

    return train_dataset
