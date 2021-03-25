import os
import math
import argparse
import sentencepiece
import torch
import pickle
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

from preprocess.lda import ETM
from preprocess.utils import load_stop_words, data_loader, build_count_vectorizer, get_nearest_neighbors
from utils.logger import init_logger, logger
from modules.optimizer import build_optim


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def optimizer_builder(model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        return optimizer


def model_builder(embedding=None, checkpoint=None):
    model = ETM(
        args.num_topics, args.vocab_size, args.hidden_size, args.rho_size, args.embed_size,
        args.device, embedding=embedding, enc_drop=args.dropout,
        train_embedding=args.train_embedding, checkpoint=checkpoint
    )
    return model


def load_embedding(embed_file, vocab):
    vocab_size = len(vocab)
    vectors = {}
    with open(embed_file, 'r') as f:
        for line in f:
            line = line.split()
            word = line[0]
            if word in vocab:
                vectors[word] = np.array(line[1:]).astype(np.float)
    embeddings = np.zeros((vocab_size, args.embed_size))
    words_found = 0
    for i, word in enumerate(vocab):
        try:
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.embed_size,))
    embeddings = torch.from_numpy(embeddings).to(args.device)
    logger.info('Found {} words / vocab size {}'.format(words_found, vocab_size))

    return embeddings


def train():
    stop_words = load_stop_words(args.stop_words_file)
    # stop_words = 'english'
    dataset = data_loader(args.data_path)
    dataset, vocab = build_count_vectorizer(dataset, stop_words, args.max_df, args.min_df)

    embedding = None if args.train_embedding else load_embedding(args.embed_file, vocab)

    logger.info('Saving vocab to {}'.format(args.model_path + '/vocab.pkl'))
    with open(args.model_path + '/vocab.pkl', 'wb') as file:
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

    model = model_builder(embedding=embedding)
    args.warmup_steps = None
    optimizer = build_optim(args, model, checkpoint=None)
    tensorboard_dir = args.model_path + '/tensorboard' + datetime.now().strftime('/%b-%d_%H-%M-%S')
    writer = SummaryWriter(tensorboard_dir)

    step = 1
    for i in range(epochs):
        model.train()
        acc_loss = 0.0
        acc_kl_theta_loss = 0.0
        for i in range(0, len_dataset, batch_size):
            batch_indices = all_indices[i: i + batch_size]
            batch = torch.tensor(dataset[batch_indices].toarray(), dtype=torch.float32, device=device)
            sums = batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = batch / sums
            else:
                normalized_data_batch = batch
            model.zero_grad()
            recon_loss, kld_theta = model(batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()
            optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()

            if step % args.report_every == 0:
                writer.add_scalar('train/loss', recon_loss, step)
                writer.add_scalar('train/kl_theta', kld_theta, step)
                writer.add_scalar('train/NELBO', recon_loss + kld_theta, step)
                logger.info('Step: {}, loss: {}, kl_theta: {}, NELBO: {} lr: {}'.format(
                    step, recon_loss, kld_theta, recon_loss + kld_theta, optimizer.learning_rate))

            step += 1
        avg_acc_loss, avg_acc_kl_theta_loss = acc_loss / epoch_steps, acc_kl_theta_loss / epoch_steps
        avg_NELBO = avg_acc_loss + avg_acc_kl_theta_loss
        logger.info('Epoch {}, avg loss: {}, avg kl_theta: {}, avg NELBO: {}'.format(
            step // epoch_steps, avg_acc_loss, avg_acc_kl_theta_loss, avg_NELBO))

        if i % 50 == 0 and i > 0:
            visualize(model, vocab)

    checkpoint = {
        'model': model.state_dict(),
        'opt': args,
        'optim': optimizer.optimizer.state_dict(),
        'num_topics': args.num_topics
    }
    checkpoint_path = os.path.join(args.model_path, 'etm.pt')
    logger.info('Saving checkpoint %s' % checkpoint_path)
    torch.save(checkpoint, checkpoint_path)


def visualize(model, vocab, show_emb=True):
    model.eval()
    queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love',
               'intelligence', 'money', 'politics', 'health', 'people', 'family']

    # visualize topics using monte carlo
    with torch.no_grad():
        logger.info('#' * 100)
        logger.info('Visualize topics...')
        topics_words = []
        gammas = model.get_beta()
        for k in range(args.num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words + 1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            logger.info('Topic {}: {}'.format(k, topic_words))

        if show_emb:
            # visualize word embeddings by using V to get nearest neighbors
            logger.info('#' * 100)
            logger.info('Visualize word embeddings by using output embedding matrix')
            try:
                embeddings = model.rho.weight  # Vocab_size x E
            except:
                embeddings = model.rho  # Vocab_size x E
            for word in queries:
                logger.info('word: {} .. neighbors: {}'.format(
                    word, get_nearest_neighbors(word, embeddings, vocab)))
            logger.info('#' * 100)


def main():
    init_logger(args.log_file)
    logger.info(args)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.spm_file)

    args.device = 'cuda' if args.use_cuda else 'cpu'

    if args.mode == 'train':
        train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/MultiNews', type=str)
    parser.add_argument('--log_file', default='../log/run_etm.log', type=str)
    parser.add_argument('--model_path', default='../models', type=str)
    parser.add_argument('--embed_file', default='../models/embeddings.txt', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--spm_file', default='../vocab/spm9998_3.model', type=str)
    parser.add_argument('--vocab_file', default='../results/etm/vocab.pt', type=str)
    parser.add_argument('--stop_words_file', default='../files/stop_words.txt', type=str)

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--report_every', default=100, type=str)

    # model-related arguments
    parser.add_argument('--num_topics', default=50, type=int)
    parser.add_argument('--rho_size', default=300, type=int)
    parser.add_argument('--embed_size', default=300, type=int)
    parser.add_argument('--hidden_size', default=800, type=int)
    parser.add_argument('--train_embedding', default=True, type=str2bool)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=2.0, type=float)
    parser.add_argument('--nonmono', default=10, type=int, help='number of bad hits allowed')
    parser.add_argument('--weight_decay', default=1.2e-6, type=float, help='some l2 regularization')
    parser.add_argument('--anneal_lr', default=False, type=str2bool, help='whether to anneal the learning rate')
    parser.add_argument('--bow_norm', default=True, type=str2bool, help='normalize the bows')

    parser.add_argument('--max_df', default=0.7, type=float)
    parser.add_argument('--min_df', default=100, type=int)

    # optimization-related arguments
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lr_scheduler', default='', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--random_seed', default=1, type=int)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.998, type=float)

    # visualization
    parser.add_argument('--num_words', default=10, type=int)

    args = parser.parse_args()

    main()
