import torch
import torch.nn as nn
import torch.nn.functional as F

from model_topic_q.neural_modules.attention_modules import TopicScaledDotProductAttention


class MultiHeadTopicAttention(nn.Module):

    def __init__(self, n_heads, d_model, dropout=0):
        super(MultiHeadTopicAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_per_head = d_model // n_heads
        self.w_qs = nn.Linear(d_model, n_heads * self.dim_per_head)
        self.w_ks = nn.Linear(d_model, n_heads * self.dim_per_head)
        self.w_vs = nn.Linear(d_model, n_heads * self.dim_per_head)
        self.topic_attn = TopicScaledDotProductAttention(d_model, self.dim_per_head, dropout)

    def forward(self, q, k, v, bias):
        """
        :param q: [batch_size, n_topic_words, d_model]
        :param k: [batch_size, tgt_len, d_model]
        :param v: [batch_size, tgt_len, d_model]
        :param bias: [batch_size, tgt_len, n_heads, n_topic_words, tgt_len]
        """
        batch_size, len_k, n_topic_words = k.size(0), k.size(1), q.size(1)
        n_heads, dim_per_head = self.n_heads, self.dim_per_head

        def shape(x):
            return x.view(batch_size, -1, n_heads, dim_per_head).transpose(1, 2)

        q, k, v = self.w_qs(q), self.w_ks(k), self.w_vs(v)
        q, k, v = shape(q), shape(k), shape(v)

        # [batch_size, n_topic_words * tgt_len, d_model]
        attn = self.topic_attn(q, k, v, bias)
        # [batch_size * tgt_len, n_topic_words, d_model]
        attn = attn.view(batch_size * len_k, -1, self.d_model)

        return attn


class TopicAttention(nn.Module):

    def __init__(self, topic_vocab_size, d_hidden):
        super(TopicAttention, self).__init__()
        self.d_hidden = d_hidden

        self.attn = nn.Linear(topic_vocab_size + d_hidden * 2, d_hidden)
        self.v = nn.Parameter(torch.FloatTensor(self.d_hidden))

    def forward(self, output, enc_output, topics):
        """
        :param output: [batch_size, n_blocks, d_hidden]
        :param enc_output: [batch_size, n_blocks, d_hidden]
        :param topics: [n_topics, topic_vocab_size]
        :return: [batch_size, n_blocks, n_topics]
        """
        batch_size, n_blocks, n_topics = enc_output.size(0), enc_output.size(1), topics.size(0)

        # [batch_size, n_topics, n_blocks, d_hidden]
        output = output.repeat(n_topics, 1, 1, 1).transpose(0, 1)
        # [batch_size, n_topics, n_blocks, d_hidden]
        enc_output = enc_output.repeat(n_topics, 1, 1, 1).transpose(0, 1)
        # [batch_size, n_topics, topic_vocab_size]
        topics = topics.repeat(batch_size, 1, 1)

        # [batch_size, n_topics, d_hidden]
        energy = torch.tanh(self.attn(torch.cat((output, enc_output, topics), dim=-1)))

        # [batch_size, d_hidden, n_topics]
        energy = energy.transpose(1, 2)

        # [batch_size, 1, d_hidden]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attn = torch.matmul(v, energy).squeeze(1)

        attn = F.softmax(attn, dim=-1)

        # [batch_size, n_topics]
        return attn
