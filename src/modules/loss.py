import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.statistics import Statistics


def build_loss_compute(symbols, vocab_size, device, train=True, label_smoothing=0.0):
    loss_compute = NMTLossCompute(symbols, vocab_size, label_smoothing=label_smoothing if train else 0.0)
    loss_compute.to(device)

    return loss_compute


class LabelSmoothingLoss(nn.Module):
    """
    通过 label smoothing 后，q (smooth 后的真实概率) 和 p (模型计算的概率) 的 KL-divergence 最小
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        :param output: [batch_size * max_tgt_len, vocab_size]
        :param target: [batch_size * max_tgt_len]
        """
        # [batch_size * max_tgt_len, vocab_size]
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(nn.Module):

    def __init__(self, symbols, vocab_size, label_smoothing=0.0):
        super(NMTLossCompute, self).__init__()
        self.padding_idx = symbols['PAD']
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_size, ignore_index=self.padding_idx)
        else:
            self.criterion = nn.NLLLoss(ignore_index=self.padding_idx, reduction='sum')

    def _stats(self, loss, scores, target):
        pred = scores.argmax(dim=1)
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _compute_loss(self, output, target):
        output = output.view(-1, output.size(1))
        ground_truth = target.contiguous().view(-1)

        loss = self.criterion(output, ground_truth)

        stats = self._stats(loss.clone(), output, ground_truth)

        return loss, stats

    def monolithic_compute_loss(self, target, output):
        _, batch_stats = self._compute_loss(output, target)
        return batch_stats

    def sharded_compute_loss(self, target, output, shard_size, normalization):
        """
        :param target: [batch_size * max_tgt_len]
        :param output: [batch_size * max_tgt_len, vocab_size]
        :param shard_size:
        :param normalization:
        """
        batch_stats = Statistics()
        shard_state = {"output": output, "target": target}
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(**shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    if eval_only:
        yield filter_shard_state(state)
    else:
        non_none = dict(filter_shard_state(state, shard_size))

        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        # grads 作为 grad_tensor (权重) 再次计算梯度
        torch.autograd.backward(inputs, grads)
