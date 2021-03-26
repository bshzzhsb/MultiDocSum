import os
import math
import argparse
import json
import glob
import gc
import sentencepiece
import torch
import pickle
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.feature_extraction.text import CountVectorizer

from preprocess.lda import ProdLDA, TopicModel
from utils.logger import init_logger, logger
from modules.optimizer import build_optim


def int_float(v):
    v = float(v)
    return int(v) if v >= 1 else v


def get_num_example():
    data_path = args.data_path
    pts = sorted(glob.glob(data_path + '/*/*.[0-9]*.json'))
    assert len(pts) > 0
    len_src, len_tgt = 0, 0
    for pt in pts:
        dataset = json.load(open(pt))
        len_tgt += len(dataset)
        for data in dataset:
            len_src += len(data['src'])

    return len_src, len_tgt


def data_loader(phase='*'):
    data_path = args.data_path

    def _lazy_dataset_loader(pt_file):
        print('loading file %s' % pt_file)
        dataset = json.load(open(pt_file))
        return dataset

    def _to_onehot(dataset, min_len):
        return np.bincount(dataset, minlength=min_len)

    pts = sorted(glob.glob(data_path + '/' + phase + '/*.[0-9]*.json'))
    assert len(pts) > 0
    np.random.shuffle(pts)

    batch = []
    for pt in pts:
        data = _lazy_dataset_loader(pt)
        for item in data:
            for src in item['src']:
                batch.append(_to_onehot(src, args.vocab_size))
                if len(batch) == args.batch_size:
                    batch = np.array(batch)
                    yield torch.tensor(batch, dtype=torch.float32)
                    batch = []

    if len(batch) > 0:
        yield torch.tensor(batch, dtype=torch.float32)


def load_stop_words():
    with open(args.stop_words_file, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file.readlines()]
        file.close()
    return stop_words


def gen_dataset(phase='*'):
    data_path = args.data_path

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


def build_count_vectorizer(dataset, stop_words):
    vectorizer = CountVectorizer(max_df=args.max_df, min_df=args.min_df, stop_words=stop_words)
    dataset = vectorizer.fit_transform(dataset)
    vocab = vectorizer.get_feature_names()

    return dataset, vocab


def optimizer_builder(model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.learning_rate, betas=(args.beta1, args.beta2)
        )
        return optimizer


def model_builder(checkpoint=None):
    model = ProdLDA(args, checkpoint=checkpoint)
    return model


def train():
    stop_words = load_stop_words()
    # stop_words = 'english'
    dataset = gen_dataset()
    dataset, vocab = build_count_vectorizer(dataset, stop_words)

    with open(args.model_path + '/vocab.pt', 'wb') as file:
        pickle.dump(vocab, file)
        file.close()

    args.vocab_size = len(vocab)
    len_dataset = dataset.shape[0]
    all_indices = list(range(len_dataset))
    np.random.shuffle(all_indices)
    batch_size, device, epochs = args.batch_size, args.device, args.epochs

    epoch_steps = math.ceil(len_dataset / batch_size)
    args.train_steps = args.epochs * epoch_steps
    logger.info('num steps: {}'.format(args.train_steps))

    model = model_builder()
    args.warmup_steps = None
    optimizer = build_optim(args, model, checkpoint=None)
    tensorboard_dir = args.model_path + '/tensorboard' + datetime.now().strftime('/%b-%d_%H-%M-%S')
    writer = SummaryWriter(tensorboard_dir)

    model.train()
    step = 1
    for _ in range(epochs):
        loss_epoch = 0.0
        for i in range(0, len_dataset, batch_size):
            batch_indices = all_indices[i: i+batch_size]
            batch = torch.tensor(dataset[batch_indices].toarray(), dtype=torch.float32, device=device)
            model.zero_grad()
            recon, loss = model(batch)
            loss.backward()
            optimizer.step()
            gc.collect()
            loss_epoch += loss.item()
            if step % args.report_every == 0:
                writer.add_scalar('train/loss', loss, step)
                logger.info('Step {}, loss {}, lr: {}'.format(step, loss, optimizer.learning_rate))
            step += 1
        logger.info('Epoch {}, average epoch loss {}'.format(step / epoch_steps, loss_epoch / epoch_steps))

    checkpoint = {
        'model': model.state_dict(),
        'opt': args,
        'optim': optimizer.optimizer.state_dict(),
        'num_topics': args.num_topics
    }
    checkpoint_path = os.path.join(args.model_path, 'prodlda_model.pt')
    logger.info('Saving checkpoint %s' % checkpoint_path)
    torch.save(checkpoint, checkpoint_path)


def test():
    assert args.checkpoint is not None

    logger.info('Loading checkpoint from %s' % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    args.num_topics = checkpoint['num_topics']

    with open(args.vocab_file, 'rb') as file:
        vocab = pickle.load(file)
        file.close()
    args.vocab_size = len(vocab)

    model = model_builder(checkpoint)
    model.eval()

    emb = model.decoder.weight.detach().numpy().T
    get_topic_words(emb, vocab)


def predict():
    assert args.checkpoint is not None

    with open(args.vocab_file, 'rb') as file:
        vocab = pickle.load(file)
        args.vocab_size = len(vocab)
        file.close()

    topic_model = TopicModel(args, vocab, args.device, args.checkpoint)
    emb = topic_model.model.decoder.weight.detach().numpy().T
    topic_words, topic_words_probs = get_topic_words(emb, vocab, 10)

    with open(args.data_path + '/train/MultiNews.30.train.11.json') as file:
        dataset = json.load(file)
        with open('../results/prod_lda/n_topic_100/topics.txt', 'w', encoding='utf-8') as out:
            for data in dataset:
                tgt = [data['tgt_str']]
                top_n_topics, top_n_topics_probs, top_n_words, top_n_words_probs = \
                    topic_model.get_topic(tgt, num_top_topic=5, num_top_word=10)
                out.write(str([(vocab[word], prob) for (word, prob) in zip(top_n_words, top_n_words_probs)]) + '\n')
                out.write(data['tgt_str'] + '\n')


def get_topic_words(beta, vocab, n_top_words=10):
    topic_words = []
    topic_words_probs = []
    logger.info('----------The Topics----------')
    for i in range(len(beta)):
        top_words = [vocab[idx] for idx in beta[i].argsort()[: -n_top_words - 1: -1]]
        top_words_probs = sorted(beta[i])[: -n_top_words - 1: -1]
        topic_words.append(top_words)
        topic_words_probs.append(top_words_probs)
        logger.info('topic {}: {}'.format(i, [_ for _ in zip(top_words, top_words_probs)]))
    logger.info('----------End of Topics----------')

    return topic_words, topic_words_probs


def main():
    init_logger(args.log_file)
    logger.info(args)
    torch.manual_seed(args.random_seed)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.spm_file)

    args.device = 'cuda' if args.use_cuda else 'cpu'

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/MultiNews', type=str)
    parser.add_argument('--log_file', default='../log/lda.log', type=str)
    parser.add_argument('--model_path', default='../models', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--spm_file', default='../vocab/spm9998_3.model', type=str)
    parser.add_argument('--vocab_file', default='../results/prod_lda/vocab.pt', type=str)
    parser.add_argument('--stop_words_file', default='../files/stop_words.txt', type=str)

    parser.add_argument('--max_df', default=0.5, type=float)
    parser.add_argument('--min_df', default=100, type=int_float)

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--report_every', default=100, type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--enc1_units', default=256, type=int)
    parser.add_argument('--enc2_units', default=256, type=int)
    parser.add_argument('--num_topics', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lr_scheduler', default='', type=str)
    parser.add_argument('--max_grad_norm', default=2.0, type=float)
    parser.add_argument('--beta1', default=0.99, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--init_mult', default=1.0, type=float)
    parser.add_argument('--variance', default=0.995, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    main()
