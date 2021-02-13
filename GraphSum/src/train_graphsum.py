import torch
import argparse
import random
import sentencepiece

from models.data_loader import Dataloader, load_dataset
from models.model import GraphSum
from models.optimizer import build_optim
from models.trainer_builder import build_trainer

from utils.logging import init_logger, logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def main(args):
    device = 'cuda' if args.use_cuda else 'cpu'
    init_logger(args.log_file)
    if args.mode == 'train':
        train(args, device)


def train(args, device):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.checkpoint != '':
        checkpoint = None
    else:
        checkpoint = None

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    # <UNK>: 0, <T>: 3, <S>: 4, </S>: 5, <PAD>: 6, <P>: 7, <Q>: 8
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'),
               'PAD': spm.PieceToId('<PAD>'), 'EOT': spm.PieceToId('<T>'),
               'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>'),
               'UNK': spm.PieceToId('<UNK>')}
    print(symbols)
    vocab_size = len(spm)

    def train_iter_fct():
        return Dataloader(args, load_dataset(args, 'train', shuffle=True), symbols,
                          args.batch_size, device, shuffle=True, is_test=False)
    test_iter = Dataloader(args, load_dataset(args, 'test', shuffle=False), symbols,
                           args.batch_size, device, shuffle=False, is_test=True)

    model = GraphSum(args, symbols['PAD'], symbols['BOS'], symbols['EOS'], tokenizer=spm)
    optim = build_optim(args, model, checkpoint)
    logger.info(model)
    trainer = build_trainer(args, device, model, symbols, vocab_size, optim, test_iter)
    trainer.train(train_iter_fct, args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_file', default='../log/graph_sum.log', type=str, help='Path to .log')
    parser.add_argument('-mode', default='train', type=str, choices=['train', 'test'],
                        help='Run mode')
    parser.add_argument('-use_cuda', default=False, type=str2bool, help='Number of gpus')
    parser.add_argument('-data_path', default='../../../data/MultiNews', type=str, help='Path to data')
    parser.add_argument('-model_path', default='../models', type=str, help='Path to save model')
    parser.add_argument('-checkpoint', default='', type=str, help='Path to checkpoint')
    parser.add_argument('-vocab_path', default='../spm/spm9998_3.model', type=str, help='Path to sentencepiece model')
    parser.add_argument('-config_path', default='../config/config.json', type=str, help='Path to config')

    parser.add_argument('-max_para_num', default=30, type=int,
                        help='Max number of paragraphs of the longest input documents')
    parser.add_argument('-max_para_len', default=60, type=int,
                        help='Max number of words in the longest paragraph')
    parser.add_argument('-max_tgt_len', default=300, type=int,
                        help='Max number of tokens in target summarization')

    parser.add_argument('-batch_size', default=4, type=int, help='Number of examples in one batch')
    parser.add_argument('-in_tokens', default=False, type=bool,
                        help='If True, batch size will be the maximum number of tokens in one batch.'
                             'else, batch size will be the maximum number of examples in one batch')

    parser.add_argument('-optim', default='adam', type=str, help='The optimizer used in training')
    parser.add_argument('-lr', default=3, type=float, help='Learning rate of the model in training')
    parser.add_argument('-max_grad_norm', default=2.0, type=float, help='The max gradient norm')
    parser.add_argument('-random_seed', default=1, type=int, help='Random seed')

    parser.add_argument('-epoch', default=20, type=int, help='Number of epochs for training')
    parser.add_argument('-save_checkpoint_steps', default=10000, help='The steps interval to save checkpoints')
    parser.add_argument('-report_every', default=100, help='The steps interval to report model')
    parser.add_argument('-val_steps', default=10000, help='The steps interval to evaluate the model performance')

    parser.add_argument('-weight_sharing', default=True, type=str2bool,
                        help='Whether to share weights between encoder and decoder')
    parser.add_argument('-max_generator_batches', default=32, type=int, help='shard size in compute loss when training')

    parser.add_argument('-max_out_len', default=300, type=int, help='max length of decoding')
    parser.add_argument('-min_out_len', default=200, type=int, help='min length of decoding')

    parser.add_argument('-beta1', default=0.9, type=float, help='Param for Adam optimizer')
    parser.add_argument('-beta2', default=0.998, type=float, help='Param for Adam optimizer')
    parser.add_argument('-warmup_steps', default=8000, type=int,
                        help='The training steps to perform linear learning rate warmup')
    parser.add_argument('-decay_method', default='noam', type=str, help='The decay method of learning rate')
    parser.add_argument('-label_smoothing', default=0.1, type=float, help='Label smoothing in loss compute')
    parser.add_argument('-pos_win', default=2.0, type=float, help='The parameter in graph attention')

    # model_config
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-enc_word_layers', default=6, type=int, help='Number of encoder word layers')
    parser.add_argument('-enc_graph_layers', default=2, type=int, help='Number of encoder graph layers')
    parser.add_argument('-dec_graph_layers', default=8, type=int, help='Number of decoder graph layers')
    parser.add_argument('-n_heads', default=8, type=int, help='Number of attention heads')
    parser.add_argument('-max_pos_embed', default=512, type=int, help='Max position embeddings')
    parser.add_argument('-hidden_act', default='relu', type=str, choices=['relu'],
                        help='Hidden activation function')
    parser.add_argument('-hidden_dropout_prob', default=0.1, type=float, help='Hidden dropout probability')
    parser.add_argument('-attn_dropout_prob', default=0.1, type=float, help='Attention Dropout probability')
    parser.add_argument('-pre_process_cmd', default='n', type=str, help='Preprocess command')
    parser.add_argument('-post_process_cmd', default='da', type=str, help='Postprocess command')
    parser.add_argument('-initializer_range', default=0.02, type=float, help='Initializer range')

    # TODO
    parser.add_argument('-beam_size', default=5, type=int, help='Beam search')
    parser.add_argument('-len_penalty', default=0.6, type=float, help='Length penalty during decoding')
    parser.add_argument('-block_trigram', default=True, type=float, help='Remove repeated trigrams in summary')

    args = parser.parse_args()

    main(args)
