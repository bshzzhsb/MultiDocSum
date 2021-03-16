import os
import math
import argparse
import json
import glob
import gc
import sentencepiece
import torch
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

from models.lda import ProdLDA
from utils.logger import init_logger, logger
from modules.optimizer import build_optim


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


def data_loader():
    data_path = args.data_path

    def _lazy_dataset_loader(pt_file):
        print('loading file %s' % pt_file)
        dataset = json.load(open(pt_file))
        return dataset

    def _to_onehot(dataset, min_len):
        return np.bincount(dataset, minlength=min_len)

    pts = sorted(glob.glob(data_path + '/*/*.[0-9]*.json'))
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


def optimizer_builder(model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.learning_rate, betas=(args.beta1, args.beta2)
        )
        return optimizer


def model_builder():
    model = ProdLDA(args)
    if args.use_cuda:
        model = model.cuda()
    return model


def train(model, gen_data_iter, optimizer):
    tensorboard_dir = args.log_path + '/tensorboard' + datetime.now().strftime('/%b-%d_%H-%M-%S')
    writer = SummaryWriter(tensorboard_dir)

    model.train()
    data_iter = gen_data_iter()
    step = 1
    while step <= args.train_steps:
        len_data = 0
        loss_epoch = 0.0
        for batch in data_iter:
            batch = batch.to(args.device)
            len_data += len(batch)
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
        epoch_steps = args.train_steps / args.epochs
        logger.info('Epoch {}, average epoch loss {}'.format(step / epoch_steps + 1, loss_epoch / epoch_steps))
        data_iter = gen_data_iter()

    checkpoint = {
        'model': model.state_dict(),
        'opt': args,
        'optim': optimizer.state_dict()
    }
    checkpoint_path = os.path.join(args.model_path, '/prodlda_model.pt')
    logger.info('Saving checkpoint %s' % checkpoint_path)
    if not os.path.exists(checkpoint_path):
        torch.save(checkpoint, checkpoint_path)


def main():
    log_dir = os.path.join(args.log_path, args.mode + datetime.now().strftime('-%b-%d_%H-%M-%S.log'))
    init_logger(log_dir)
    logger.info(args)
    torch.manual_seed(args.random_seed)

    args.device = 'cuda' if args.use_cuda else 'cpu'
    epoch_steps = math.ceil(get_num_example()[0] / args.batch_size)
    args.train_steps = args.epochs * epoch_steps
    logger.info('num steps: {}'.format(args.train_steps))

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    args.vocab_size = len(spm)

    model = model_builder()
    args.warmup_steps = None
    optimizer = build_optim(args, model, checkpoint=None)
    train(model, data_loader, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/MultiNews', type=str)
    parser.add_argument('--vocab_path', default='../spm/spm9998_3.model', type=str)
    parser.add_argument('--log_path', default='../log', type=str)
    parser.add_argument('--model_path', default='../models', type=str)

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--report_every', default=100, type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--enc1_units', default=256, type=int)
    parser.add_argument('--enc2_units', default=256, type=int)
    parser.add_argument('--num_topic', default=50, type=int)
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
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    main()
