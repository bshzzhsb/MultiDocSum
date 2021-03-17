import json
import glob
import numpy as np
import sentencepiece
import pickle
import argparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.logger import init_logger, logger


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


def train_sklearn_lda(data_path, spm):
    dataset = load_dataset(data_path, False)

    lda = LatentDirichletAllocation(n_components=20, verbose=1)

    train_set = []
    for batch in dataset:
        for data in batch:
            for item in data['src']:
                train_set.append(spm.DecodeIdsWithCheck(item))

    vectorizer = CountVectorizer(max_df=args.max_df, min_df=args.min_df)
    train_set = vectorizer.fit_transform(train_set)
    vocab = vectorizer.get_feature_names()

    lda.fit(train_set)

    topic_word = lda.components_
    with open('./topic_word.pt', 'wb') as file:
        pickle.dump(topic_word, file)
        file.close()
    n_topic_word = 20

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_topic_word + 1):-1]
        logger.info('Topic %d : %s' % (i, ' '.join(topic_words)))


def main():
    data_path = args.data_path
    logger_file = args.log_file
    vocab_path = args.vocab_path

    init_logger(logger_file)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(vocab_path)
    vocab_size = len(spm)

    train_sklearn_lda(data_path, spm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/MultiNews', type=str)
    parser.add_argument('--vocab_path', default='../spm/spm9998_3.model', type=str)
    parser.add_argument('--log_file', default='../log/lda.log', type=str)
    parser.add_argument('--model_path', default='../models', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--max_df', default=0.3, type=float)
    parser.add_argument('--min_df', default=0.001, type=float)
    args = parser.parse_args()
    main()

    # from utils.logger import init_logger, logger
    # from utils.cal_rouge import test_rouge, rouge_results_to_str
    #
    # init_logger('../log/predict_torch_sc_my.log')
    #
    # candi = open('../results/graph_sum/res.candidate')
    # ref = open('../results/graph_sum/res.gold')
    # result_dict = test_rouge(candi, ref, 1)
    #
    # logger.info('Rouges %s' % (rouge_results_to_str(result_dict)))

    # pred_file = open('../results/graph_sum/official_paddle.candidate', 'r', encoding='utf-8')
    # gold_file = open('../results/graph_sum/official_paddle.gold', 'r', encoding='utf-8')
    #
    # out_pred_file = open('../results/graph_sum/official.candidate', 'w', encoding='utf-8')
    # out_gold_file = open('../results/graph_sum/official.gold', 'w', encoding='utf-8')
    #
    # preds = pred_file.readlines()
    # golds = gold_file.readlines()
    #
    # for pred, gold in zip(preds, golds):
    #     pred_str = pred.replace('<q>', ' ').replace(' +', ' ') \
    #         .replace('<unk>', 'UNK').replace('\\', '').strip()
    #     gold_str = gold.replace('<t>', '').replace('</t>', '') \
    #         .replace('<q>', ' ').replace(' +', ' ').replace('\\', '').strip()
    #
    #     out_pred_file.write(pred_str + '\n')
    #     out_gold_file.write(gold_str + '\n')
    #
    # pred_file.close()
    # gold_file.close()
    # out_pred_file.close()
    # out_gold_file.close()

    # pred_file = open('../results/graph_sum/sc_pos_embed_lr_3_step_160000/results/res.160000.candidate',
    #                  'r', encoding='utf-8')
    # gold_file = open('../results/graph_sum/sc_pos_embed_lr_3_step_160000/results/res.160000.gold',
    #                  'r', encoding='utf-8')
    # out_pred_file = open('../results/graph_sum/res.candidate', 'w', encoding='utf-8')
    # out_gold_file = open('../results/graph_sum/res.gold', 'w', encoding='utf-8')
    # preds = pred_file.readlines()
    # golds = gold_file.readlines()
    #
    # for pred, gold in zip(preds, golds):
    #     pred_str = pred.replace(' ##', '').replace('<S>', '').replace('</S>', '').replace('<Q>', '<q>') \
    #         .replace('<P>', ' ').replace('<T>', '').replace('<PAD>', '').replace(' ⁇ ', ''). \
    #         replace('⁇ ', '').replace(' ⁇', '').strip(' \n"')
    #     gold_str = gold.replace(' ##', '').replace('<S>', '').replace('</S>', '').replace('<Q>', '<q>') \
    #         .replace('<P>', ' ').replace('<T>', '').replace('<PAD>', '').replace(' ⁇ ', ''). \
    #         replace('⁇ ', '').replace(' ⁇', '').strip(' \n"')
    #     out_pred_file.write(pred_str + '\n')
    #     out_gold_file.write(gold_str + '\n')
    #
    # pred_file.close()
    # gold_file.close()
    # out_pred_file.close()
    # out_gold_file.close()
