import json
import glob
import gc

import numpy as np
import torch

from utils.logger import logger


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]


class DataBatch(object):

    def __init__(self, n_heads, max_para_num, max_para_len, max_tgt_len, n_topic_words,
                 data=None, pad_idx=None, device=None, is_test=False):
        self.n_heads = n_heads
        self.max_para_num = max_para_num
        self.max_para_len = max_para_len
        self.max_tgt_len = max_tgt_len
        self.n_topic_words = n_topic_words
        self.pad_idx = pad_idx
        self.device = device

        if data is not None:
            # src, tgt_ids, label_ids, tgt_str, graph
            self.batch_size = len(data)

            enc_input, dec_input, tgt_label, label_weight = self.process_batch(data, device)

            setattr(self, 'enc_input', enc_input)
            setattr(self, 'dec_input', dec_input)
            setattr(self, 'tgt_label', tgt_label)
            setattr(self, 'label_weight', label_weight)

            if is_test:
                tgt_str = [inst[3] for inst in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size

    def process_batch(self, data, device):
        src_words, src_words_pos, src_sents_pos, src_words_self_attn_bias, \
            src_sents_self_attn_bias, graph_attn_bias = self._pad_src_batch_data(
                insts=[inst[0] for inst in data],
                graphs=[inst[4] for inst in data],
                device=device
            )

        tgt_words, tgt_pos, tgt_self_attn_bias = self._pad_tgt_batch_data(
            insts=[inst[1] for inst in data],
            device=device
        )

        tgt_label, label_weight = self._pad_label_batch_data(
            insts=[inst[2] for inst in data],
            device=device
        )

        # [batch_size, max_para_num, n_heads, max_para_len, max_para_len]
        src_words_self_attn_bias = src_words_self_attn_bias.unsqueeze(2).unsqueeze(3) \
            .expand(-1, -1, self.n_heads, self.max_para_len, -1)
        src_words_self_attn_bias.requires_grad = False

        src_sents_self_attn_bias = src_sents_self_attn_bias.unsqueeze(1).unsqueeze(2) \
            .expand(-1, self.n_heads, self.max_para_num, -1)
        src_sents_self_attn_bias.requires_grad = False

        graph_attn_bias = graph_attn_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        graph_attn_bias.requires_grad = False

        tgt_self_attn_bias = tgt_self_attn_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        tgt_self_attn_bias.requires_grad = False

        # [batch_size, max_para_num, n_heads, max_tgt_len, max_para_len]
        tgt_src_words_attn_bias = src_words_self_attn_bias[:, :, :, 0].unsqueeze(3) \
            .expand(-1, -1, -1, self.max_tgt_len, -1)
        tgt_src_words_attn_bias.requires_grad = False

        tgt_src_sents_attn_bias = src_sents_self_attn_bias[:, :, 0].unsqueeze(2) \
            .expand(-1, -1, self.max_tgt_len, -1)
        tgt_src_sents_attn_bias.requires_grad = False

        src_words = src_words.view(-1, self.max_para_num, self.max_para_len)
        src_words_pos = src_words_pos.view(-1, self.max_para_num, self.max_para_len)
        src_sents_pos = src_sents_pos.view(-1, self.max_para_num)
        tgt_words = tgt_words.view(-1, self.max_tgt_len)
        tgt_pos = tgt_pos.view(-1, self.max_tgt_len)
        tgt_label = tgt_label.view(-1, 1)
        label_weight = label_weight.view(-1, 1)

        tgt_topic, tgt_topic_attn_bias, para_topic, para_topic_attn_bias = self._pad_topic_batch_data(
            tgt_topic_insts=[inst[5] for inst in data],
            para_topic_insts=[inst[6] for inst in data]
        )
        tgt_topic_attn_bias = tgt_topic_attn_bias.unsqueeze(1).unsqueeze(2)\
            .expand(-1, self.n_heads, self.max_tgt_len, -1)
        para_topic_attn_bias = para_topic_attn_bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        enc_input = (src_words, src_words_pos, src_sents_pos, src_words_self_attn_bias,
                     src_sents_self_attn_bias, graph_attn_bias)
        dec_input = (tgt_words, tgt_pos, tgt_self_attn_bias, tgt_src_words_attn_bias,
                     tgt_src_sents_attn_bias, graph_attn_bias, tgt_topic, tgt_topic_attn_bias,
                     para_topic, para_topic_attn_bias)

        return enc_input, dec_input, tgt_label, label_weight

    def _pad(self, data, height, width, pad_id):
        # input => [height, width]
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))

        return rtn_data

    def _pad_src_batch_data(self, insts, graphs, device):
        # [batch_size, n_blocks, n_tokens]
        src_words = [self._pad(inst, self.max_para_num, self.max_para_len, self.pad_idx)
                     for inst in insts]
        src_words = torch.tensor(src_words, dtype=torch.int64, device=device)

        # [batch_size, n_blocks, n_tokens]
        src_words_pos = [[list(range(0, len(para))) + [0] * (self.max_para_len - len(para))
                          for para in inst] +
                         [[0] * self.max_para_len] * (self.max_para_num - len(inst))
                         for inst in insts]
        src_words_pos = torch.tensor(src_words_pos, dtype=torch.int64, device=device)

        # [batch_size, n_blocks]
        src_sents_pos = [list(range(0, len(inst))) + [0] * (self.max_para_num - len(inst))
                         for inst in insts]
        src_sents_pos = torch.tensor(src_sents_pos, dtype=torch.int64, device=device)

        # 在 paddings 上不计算 attention
        # [batch_size, n_blocks, n_tokens]
        src_words_self_attn_bias = [[[0.0] * len(para) + [-1e18] * (self.max_para_len - len(para))
                                     for para in inst] +
                                    [[-1e18] * self.max_para_len] * (self.max_para_num - len(inst))
                                    for inst in insts]
        src_words_self_attn_bias = torch.tensor(src_words_self_attn_bias, dtype=torch.float32, device=device)

        # [batch_size, n_blocks]
        src_sents_self_attn_bias = [[0.0] * len(inst) + [-1e18] * (self.max_para_num - len(inst))
                                    for inst in insts]
        src_sents_self_attn_bias = torch.tensor(src_sents_self_attn_bias, dtype=torch.float32, device=device)

        graphs = [[[1.0 - float(sim) for sim in list(row)] for row in g] for g in graphs]
        # [batch_size, n_blocks, n_blocks]
        graph_attn_bias = [self._pad(g, self.max_para_num, self.max_para_num, 1.0) for g in graphs]
        graph_attn_bias = torch.tensor(graph_attn_bias, dtype=torch.float32, device=device)

        return [src_words, src_words_pos, src_sents_pos, src_words_self_attn_bias,
                src_sents_self_attn_bias, graph_attn_bias]

    def _pad_tgt_batch_data(self, insts, device):
        # [batch_size, max_tgt_len]
        tgt_words = [inst + [self.pad_idx] * (self.max_tgt_len - len(inst))
                     for inst in insts]
        tgt_words = torch.tensor(tgt_words, dtype=torch.int64, device=device)

        # [batch_size, max_tgt_len]
        tgt_pos = [list(range(0, len(inst))) + [0] * (self.max_tgt_len - len(inst))
                   for inst in insts]
        tgt_pos = torch.tensor(tgt_pos, dtype=torch.int64, device=device)

        # tgt_self_attn_bias = [[[1.0] * self.max_tgt_len] * self.max_tgt_len] * len(insts)
        tgt_self_attn_pad_bias = [[0.0] * len(inst) + [1.0] * (self.max_tgt_len - len(inst)) for inst in insts]
        tgt_self_attn_pad_bias = torch.tensor(tgt_self_attn_pad_bias, dtype=torch.bool, device=device).unsqueeze(-2)
        # 上三角矩阵
        # [batch_size, max_tgt_len, max_tgt_len]
        tgt_self_attn_subsequent_bias = torch.triu(
            torch.ones([len(insts), self.max_tgt_len, self.max_tgt_len], dtype=torch.float, device=device),
            diagonal=1
        ).to(torch.bool)
        tgt_self_attn_bias = (tgt_self_attn_pad_bias | tgt_self_attn_subsequent_bias).to(torch.float) * -1e18
        # * -1e18

        return [tgt_words, tgt_pos, tgt_self_attn_bias]

    def _pad_label_batch_data(self, insts, device):
        # [batch_size, max_tgt_len]
        tgt_label = [inst + [self.pad_idx] * (self.max_tgt_len - len(inst))
                     for inst in insts]
        tgt_label = torch.tensor(tgt_label, dtype=torch.int64, device=device)

        # [batch_size, max_tgt_len]
        label_weight = [[1.0] * len(inst) + [0.0] * (self.max_tgt_len - len(inst))
                        for inst in insts]
        label_weight = torch.tensor(label_weight, dtype=torch.float32, device=device)

        return [tgt_label, label_weight]

    def _pad_topic_batch_data(self, tgt_topic_insts, para_topic_insts):
        tgt_topic = [inst + [self.pad_idx] * (self.n_topic_words - len(inst))
                     for inst in tgt_topic_insts]
        tgt_topic = torch.tensor(tgt_topic, dtype=torch.int64, device=self.device)

        tgt_topic_attn_bias = [[0.0] * len(inst) + [-1e18] * (self.n_topic_words - len(inst))
                               for inst in tgt_topic_insts]
        tgt_topic_attn_bias = torch.tensor(tgt_topic_attn_bias, dtype=torch.float, device=self.device)

        para_topic = [inst + [self.pad_idx] * (self.max_para_num - len(inst))
                      for inst in para_topic_insts]
        para_topic = torch.tensor(para_topic, dtype=torch.int64, device=self.device)

        para_topic_attn_bias = [[[0.0] * len(tgt_topic_inst) + [-1e18] * (self.n_topic_words - len(tgt_topic_inst))]
                                * len(para_topic_inst) +
                                [[-1e18] * self.n_topic_words] * (self.max_para_num - len(para_topic_inst))
                                for (para_topic_inst, tgt_topic_inst) in zip(para_topic_insts, tgt_topic_insts)]
        para_topic_attn_bias = torch.tensor(para_topic_attn_bias, dtype=torch.float, device=self.device)

        return tgt_topic, tgt_topic_attn_bias, para_topic, para_topic_attn_bias


def load_dataset(args, phase, shuffle):
    assert phase in ['train', 'valid', 'test']

    def _lazy_dataset_loader(pt_file, phase):
        dataset = json.load(open(pt_file))
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (phase, pt_file, len(dataset)))
        return dataset

    pts = sorted(glob.glob(args.data_path + '/' + phase + '/*.[0-9]*.json'))
    if pts:
        if shuffle:
            np.random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, phase)
    else:
        pt = sorted(glob.glob(args.data_path + '/' + phase + '/*.json'))
        yield _lazy_dataset_loader(pt, phase)


def get_num_examples(data_path, phase):
    assert phase in ['train', 'valid', 'test']

    def _lazy_dataset_loader(pt_file):
        dataset = json.load(open(pt_file))
        return len(dataset)

    num = 0
    pts = sorted(glob.glob(data_path + '/' + phase + '/*.[0-9]*.json'))
    if pts:
        for pt in pts:
            num += _lazy_dataset_loader(pt)
    else:
        pt = sorted(glob.glob(data_path + '/' + phase + '/*.json'))
        num += _lazy_dataset_loader(pt)
    return num


class DataLoader(object):

    def __init__(self, args, datasets, symbols, batch_size, device,
                 shuffle, is_test, random_seed=None):
        self.args = args
        self.datasets = datasets
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

        if not random_seed:
            random_seed = 0
        np.random.seed(random_seed)

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            if hasattr(self, 'cur_dataset'):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args, dataset=self.cur_dataset, symbols=self.symbols,
                            batch_size=self.batch_size, device=self.device,
                            is_test=self.is_test, shuffle=self.shuffle)


class DataIterator(object):

    def __init__(self, args, dataset, symbols, batch_size, graph_type='similarity',
                 device=None, is_test=False, shuffle=True):
        self.args = args
        self.max_para_num = self.args.max_para_num
        self.max_para_len = self.args.max_para_len
        self.max_tgt_len = self.args.max_tgt_len
        self.topic_threshold = self.args.topic_threshold
        self.min_topic_words = self.args.min_topic_words

        assert 0 <= self.min_topic_words <= 10

        self.dataset = dataset
        self.batch_size = batch_size
        self.graph_type = graph_type
        self.device = device
        self.is_test = is_test
        self.shuffle = shuffle

        self.symbols = symbols
        self.eos_idx = self.symbols['EOS']

        assert self.graph_type == 'similarity'

        self.iterations = 0

        self.secondary_sort_key = lambda x: sum([len(xi) for xi in x[0]])
        self.primary_sort_key = lambda x: len(x[1])
        self._iterations_this_epoch = 0

    def data(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex):
        src, tgt, tgt_str, graph, tgt_topic, para_topic = \
            ex['src'], ex['tgt'], ex['tgt_str'], ex['sim_graph'], ex['tgt_topic'], ex['src_topic']

        src = src[:self.max_para_num]
        src = [para[:self.max_para_len] for para in src]

        graph = graph[:self.max_para_num]
        graph = [sim[:self.max_para_num] for sim in graph]

        tgt = tgt[:-1][:self.max_tgt_len] + [self.eos_idx]
        tgt_ids = tgt[:-1]
        label_ids = tgt[1:]

        if tgt_topic[2][1] > self.topic_threshold:
            tgt_topic = [topic[0] for topic in tgt_topic if topic[1] >= self.topic_threshold]
        else:
            tgt_topic = [topic[0] for topic in tgt_topic][:self.min_topic_words]

        return src, tgt_ids, label_ids, tgt_str, graph, tgt_topic, para_topic

    def simple_batch_size_fn(self, new, count):
        src, tgt = new[0], new[1]
        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
        max_src_in_batch = max(max_src_in_batch, len(src))
        src_elements = count * max_src_in_batch
        return src_elements

    def get_batch(self, data, batch_size):
        batch, max_len = [], 0
        for ex in data:
            max_len = max(max_len, len(ex[1]))
            if self.args.in_tokens:
                to_append = (len(batch) + 1) * max_len <= batch_size
            else:
                to_append = len(batch) < batch_size
            if to_append:
                batch.append(ex)
            else:
                yield batch
                batch, max_len = [ex], len([ex[1]])
        if batch:
            yield batch

    def batch_buffer(self, data, batch_size):
        batch, max_len = [], 0
        for ex in data:
            ex = self.preprocess(ex)
            max_len = max(max_len, len(ex[1]))
            if self.args.in_tokens:
                to_append = (len(batch) + 1) * max_len <= batch_size
            else:
                to_append = len(batch) < batch_size
            if to_append:
                batch.append(ex)
            else:
                yield batch
                batch, max_len = [ex], len(ex[1])

        if batch:
            yield batch

    def create_batches(self):
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 100):
            if self.args.mode != 'train':
                p_batch = self.get_batch(
                    buffer,
                    self.batch_size
                )
            else:
                p_batch = self.get_batch(
                    sorted(sorted(buffer, key=self.secondary_sort_key), key=self.primary_sort_key),
                    self.batch_size
                )
            # list 可以迭代完 get_batch
            p_batch = list(p_batch)

            if self.shuffle:
                np.random.shuffle(p_batch)
            for batch in p_batch:
                if len(batch) == 0:
                    continue
                yield batch

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, mini_batch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = DataBatch(self.args.n_heads, self.args.max_para_num, self.args.max_para_len,
                                  self.args.max_tgt_len, self.args.num_topic_words,
                                  mini_batch, self.symbols['PAD'], self.device, self.is_test)

                yield batch
            return
