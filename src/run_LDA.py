import os
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
    step = 0
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
            loss_epoch += loss
            if step % args.report_every == 0:
                writer.add_scalar('train/loss', loss, step)
                logger.info('Step {}, loss {}, lr: {}'.format(step, loss, optimizer.learning_rate))
            step += 1
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
    log_dir = args.log_path + '/' + args.mode + datetime.now().strftime('-%b-%d_%H-%M-%S.log')
    init_logger(log_dir)
    logger.info(args)
    torch.manual_seed(args.random_seed)

    args.device = 'cuda' if args.use_cuda else 'cpu'
    # epoch_steps = math.ceil(get_num_example()[0] / args.batch_size)
    epoch_steps = 20000
    args.train_steps = args.epochs * epoch_steps
    args.warmup_steps = args.train_steps * 0.1

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    args.vocab_size = len(spm)

    model = model_builder()
    optimizer = build_optim(args, model, checkpoint=None)
    train(model, data_loader, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data/MultiNews')
    parser.add_argument('--vocab_path', type=str, default='../spm/spm9998_3.model')
    parser.add_argument('--log_path', type=str, default='../log')
    parser.add_argument('--model_path', type=str, default='../models')

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--report_every', type=str, default=100)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--enc1_units', type=int, default=256)
    parser.add_argument('--enc2_units', type=int, default=256)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_scheduler', type=str, default='linear_warmup_decay')
    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument('--beta1', type=float, default=0.99)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--init_mult', type=float, default=1.0)
    parser.add_argument('--variance', type=float, default=0.995)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    main()
