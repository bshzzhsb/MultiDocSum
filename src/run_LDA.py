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

from preprocess.lda import ProdLDA, TopicModel
from preprocess.utils import data_loader, load_stop_words, build_count_vectorizer, get_count_vectorizer
from utils.logger import init_logger, logger


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


def optimizer_builder(model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, betas=(args.beta1, args.beta2)
        )
        return optimizer


def model_builder(checkpoint=None):
    model = ProdLDA(args.num_topics, args.enc1_units, args.enc2_units, args.vocab_size, args.variance,
                    args.dropout, args.device, args.init_mult, checkpoint=checkpoint)
    return model


def train(spm):
    vocab_save_file = args.model_path + '/vocab.pkl'
    model_save_file = args.model_path + '/prodlda_model.pt'

    stop_words = load_stop_words(args.stop_words_file)
    # stop_words = 'english'
    dataset = data_loader(args.data_path, source=args.source, spm=spm)
    if args.source == 'tgt':
        dataset, vocab = build_count_vectorizer(dataset, stop_words, args.max_df, args.min_df)
        logger.info('Saving vocab to {}, vocab size {}'.format(vocab_save_file, len(vocab)))
        with open(vocab_save_file, 'wb') as file:
            pickle.dump(vocab, file)
            args.vocab_size = len(vocab)
    else:
        with open(args.vocab_file, 'rb') as file:
            vocab = pickle.load(file)
            args.vocab_size = len(vocab)
        logger.info('Loaded vocab {}, vocab size {}'.format(args.vocab_file, len(vocab)))
        dataset = get_count_vectorizer(dataset, vocab)

    len_dataset = dataset.shape[0]
    all_indices = list(range(len_dataset))
    np.random.shuffle(all_indices)
    batch_size, device, epochs = args.batch_size, args.device, args.epochs

    epoch_steps = math.ceil(len_dataset / batch_size)
    args.train_steps = args.epochs * epoch_steps
    logger.info('num steps: {}'.format(args.train_steps))

    model = model_builder()
    args.warmup_steps = None
    optimizer = optimizer_builder(model)
    tensorboard_dir = args.model_path + '/tensorboard' + datetime.now().strftime('/%b-%d_%H-%M-%S')
    writer = SummaryWriter(tensorboard_dir)

    model.train()
    step = 0
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
            if step % args.report_every == 0 and step > 0:
                writer.add_scalar('train/loss', loss, step)
                logger.info('Step {}, loss {}, lr: {}'.format(step, loss, optimizer.param_groups[0]['lr']))
            step += 1
        logger.info('Epoch {}, average epoch loss {}'.format(step // epoch_steps, loss_epoch / epoch_steps))

    checkpoint = {
        'model': model.state_dict(),
        'opt': args,
        'optim': optimizer.state_dict(),
        'num_topics': args.num_topics
    }
    checkpoint_path = os.path.join(model_save_file)
    logger.info('Saving checkpoint %s' % checkpoint_path)
    torch.save(checkpoint, checkpoint_path)


def train_src(spm):
    model_save_file = args.model_path + '/prodlda_model.pt'

    dataset = data_loader(args.data_path, source=args.source, spm=spm)
    with open(args.vocab_file, 'rb') as file:
        vocab = pickle.load(file)
        args.vocab_size = len(vocab)
    logger.info('Loaded vocab {}, vocab size {}'.format(args.vocab_file, len(vocab)))
    vectorizer = get_count_vectorizer(vocab)

    len_dataset = len(dataset)
    batch_size, device, epochs = args.batch_size, args.device, args.epochs

    epoch_steps = math.ceil(len_dataset / batch_size)
    args.train_steps = args.epochs * epoch_steps
    logger.info('num steps: {}'.format(args.train_steps))

    model = model_builder()
    args.warmup_steps = None
    optimizer = optimizer_builder(model)
    tensorboard_dir = args.model_path + '/tensorboard' + datetime.now().strftime('/%b-%d_%H-%M-%S')
    writer = SummaryWriter(tensorboard_dir)

    model.train()
    step = 0
    for _ in range(epochs):
        loss_epoch = 0.0
        np.random.shuffle(dataset)
        for i in range(0, len_dataset, batch_size):
            batch = vectorizer.transform(dataset[i: i + batch_size]).toarray()
            batch = torch.tensor(batch, dtype=torch.float32, device=device)
            model.zero_grad()
            recon, loss = model(batch)
            loss.backward()
            optimizer.step()
            gc.collect()
            loss_epoch += loss.item()
            if step % args.report_every == 0 and step > 0:
                writer.add_scalar('train/loss', loss, step)
                logger.info('Step {}, loss {}, lr: {}'.format(step, loss, optimizer.param_groups[0]['lr']))
            step += 1
        logger.info('Epoch {}, average epoch loss {}'.format(step // epoch_steps, loss_epoch / epoch_steps))

    checkpoint = {
        'model': model.state_dict(),
        'opt': args,
        'optim': optimizer.state_dict(),
        'num_topics': args.num_topics
    }
    checkpoint_path = os.path.join(model_save_file)
    logger.info('Saving checkpoint %s' % checkpoint_path)
    torch.save(checkpoint, checkpoint_path)


def predict(spm):
    assert args.checkpoint is not None

    with open(args.vocab_file, 'rb') as file:
        vocab = pickle.load(file)
        args.vocab_size = len(vocab)

    topic_model = TopicModel(vocab, args.device, args.checkpoint)
    emb = topic_model.model.decoder.weight.detach().numpy().T
    topic_words, topic_words_probs = get_topic_words(emb, vocab, 10)

    with open(args.data_path + '/test/MultiNews.30.test.1.json') as file:
        dataset = json.load(file)
        with open('../results/prodlda/topics.txt', 'w', encoding='utf-8') as out:
            for data in dataset:
                # srcs = [[data['tgt_str']]]
                srcs = [spm.DecodeIds(src) for src in data['src']]
                top_n_words, top_n_words_probs = \
                    topic_model.get_srcs_topic_words(srcs, n_topic_words=10)
                out.write(str([(vocab[word], prob) for (word, prob) in zip(top_n_words, top_n_words_probs)]) + '\n')
                out.write(srcs[0] + '\n')


def preprocess(spm):
    assert args.checkpoint is not None

    with open(args.vocab_file, 'rb') as file:
        vocab = pickle.load(file)
        args.vocab_size = len(vocab)

    topic_model = TopicModel(vocab, args.device, args.checkpoint)

    phases = ['train', 'test', 'dev']

    for phase in phases:
        pts = sorted(glob.glob(args.data_path + '/' + phase + '/*.[0-9]*.json'))
        assert len(pts) > 0
        for pt in pts:
            with open(pt) as pt_file:
                print('loading file %s' % pt)
                data = json.load(pt_file)
                pt_name = pt_file.name.split('/')[-1]
            with open(args.out_path + '/' + phase + '/' + pt_name, 'w', encoding='utf-8') as out_file:
                if args.source == 'tgt':
                    for i, item in enumerate(data):
                        tgt = [item['tgt_str']]
                        top_n_words, top_n_words_probs = \
                            topic_model.get_topic_words(tgt, n_topic_words=10)
                        top_n_words = [vocab[idx] for idx in top_n_words]
                        top_n_words_ids = [idx[0] for idx in spm.Encode(top_n_words)]
                        data[i]['tgt_topic'] = [(word_id, prob)
                                                for (word_id, prob) in zip(top_n_words_ids, top_n_words_probs.tolist())]
                else:
                    for i, item in enumerate(data):
                        data[i]['src_topic'] = []
                        for j, src in enumerate(item['src']):
                            src = [spm.DecodeIds(src)]
                            top_n_words, top_n_words_probs = \
                                topic_model.get_topic_words(src, n_topic_words=1)
                            top_n_words = [vocab[idx] for idx in top_n_words]
                            top_n_words_ids = [idx[0] for idx in spm.Encode(top_n_words)]
                            data[i]['src_topic'].append(top_n_words_ids[0])
                json.dump(data, out_file)


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
        if args.source == 'tgt':
            train(spm)
        elif args.source == 'src':
            train_src(spm)
    elif args.mode == 'predict':
        predict(spm)
    elif args.mode == 'preprocess':
        preprocess(spm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/MultiNews', type=str)
    parser.add_argument('--log_file', default='../log/lda.log', type=str)
    parser.add_argument('--model_path', default='../models', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--spm_file', default='../vocab/spm9998_3.model', type=str)
    parser.add_argument('--vocab_file', default='../results/prod_lda/vocab.pkl', type=str)
    parser.add_argument('--stop_words_file', default='../files/stop_words.txt', type=str)
    parser.add_argument('--out_path', default='../../data/MultiNewsTopic', type=str)

    parser.add_argument('--max_df', default=0.5, type=float)
    parser.add_argument('--min_df', default=100, type=int_float)

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--source', default='tgt', type=str)
    parser.add_argument('--report_every', default=100, type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--enc1_units', default=256, type=int)
    parser.add_argument('--enc2_units', default=256, type=int)
    parser.add_argument('--num_topics', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lr_scheduler', default='', type=str)
    parser.add_argument('--max_grad_norm', default=2.0, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.998, type=float)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--init_mult', default=1.0, type=float)
    parser.add_argument('--variance', default=0.995, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    main()
