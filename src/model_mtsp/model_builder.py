import torch
import torch.nn as nn
from torch.nn.init import normal_

from models.layers.encoder import TransformerEncoder, GraphEncoder
from model_mtsp.neural_modules.decoder import GraphDecoder
from models.neural_modules.neural_modules import PositionalEncoding


class MDSTopicSP(nn.Module):

    def __init__(self, args, symbols, tokenizer, device, checkpoint=None):
        super(MDSTopicSP, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.batch_size = args.batch_size
        self.max_para_num = args.max_para_num
        self.max_para_len = args.max_para_len
        self.max_tgt_len = args.max_tgt_len
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.max_pos_len = args.max_pos_embed
        self.embed_size = args.hidden_size
        self.padding_idx = symbols['PAD']
        self.bos_idx = symbols['BOS']
        self.eos_idx = symbols['EOS']

        self.d_model = self.embed_size
        self.weight_sharing = args.weight_sharing
        self.pos_win = args.pos_win
        self.enc_word_layers = args.enc_word_layers
        self.enc_graph_layers = args.enc_graph_layers
        self.dec_graph_layers = args.dec_graph_layers
        self.n_heads = args.n_heads
        self.dropout = args.dropout_prob
        self.initializer_std = args.initializer_range

        self.beam_size = args.beam_size
        self.block_trigram = args.block_trigram

        self.enc_word_embed = nn.Embedding(self.vocab_size, self.embed_size, self.padding_idx)
        self.enc_embed_dropout = nn.Dropout(self.dropout)
        self.dec_embed = nn.Embedding(self.vocab_size, self.embed_size, self.padding_idx)
        self.dec_embed_dropout = nn.Dropout(self.dropout)
        self.enc_pos_embed = PositionalEncoding(self.embed_size // 2)
        self.dec_pos_embed = PositionalEncoding(self.embed_size)

        self.tgt_topic_embed = nn.Embedding(self.vocab_size, self.embed_size, self.padding_idx)
        self.para_topic_embed = nn.Embedding(self.vocab_size, self.embed_size, self.padding_idx)

        if self.weight_sharing:
            self.dec_embed.weight = self.enc_word_embed.weight
            self.tgt_topic_embed.weight = self.enc_word_embed.weight
            self.para_topic_embed.weight = self.enc_word_embed.weight

        self.transformer_encoder = TransformerEncoder(
            n_layers=self.enc_word_layers,
            n_heads=self.n_heads,
            d_k=self.embed_size // self.n_heads,
            d_v=self.embed_size // self.n_heads,
            d_model=self.embed_size,
            d_inner_hidden=self.embed_size * 4,
            dropout=self.dropout
        )
        self.graph_encoder = GraphEncoder(
            n_graph_layers=self.enc_graph_layers,
            n_heads=self.n_heads,
            n_blocks=self.max_para_num,
            d_model=self.embed_size,
            d_k=self.embed_size // self.n_heads,
            d_v=self.embed_size // self.n_heads,
            d_inner_hidden=self.embed_size * 4,
            pos_win=self.pos_win,
            dropout=self.dropout
        )
        self.enc_layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.graph_decoder = GraphDecoder(
            n_layers=self.dec_graph_layers,
            n_heads=self.n_heads,
            d_model=self.embed_size,
            d_k=self.embed_size // self.n_heads,
            d_v=self.embed_size // self.n_heads,
            d_inner_hidden=self.embed_size * 4,
            pos_win=self.pos_win,
            dropout=self.dropout,
            device=device,
        )

        self.generator_fc = nn.Linear(self.embed_size, self.vocab_size)
        if self.weight_sharing:
            self.generator_fc.weight = self.dec_embed.weight
            self.generator_fc.bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float), requires_grad=True)

        self.generator_log_softmax = nn.LogSoftmax(dim=-1)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    normal_(p, mean=0, std=self.initializer_std)
                    # xavier_uniform_(p)

        self.to(device)

    def encode(self, enc_input):
        src_word, src_word_pos, src_sent_pos, src_words_self_attn_bias, \
            src_sent_self_attn_bias, graph_attn_bias = enc_input

        # [batch_size, n_blocks, n_tokens, d_model]
        word_embed_out = self.enc_word_embed(src_word)
        word_embed_out = word_embed_out * (self.embed_size ** 0.5)

        # [batch_size, n_blocks, n_tokens, d_model / 2]
        word_pos_out = self.enc_pos_embed(src_word_pos)

        # [batch_size, n_blocks, d_model / 2]
        sent_pos_out = self.enc_pos_embed(src_sent_pos)

        # [batch_size, n_blocks, n_tokens, d_model / 2]
        sent_pos_out = torch.unsqueeze(sent_pos_out, 2).expand(-1, -1, self.max_para_len, -1)

        # [batch_size, n_blocks, n_tokens, d_model]
        combined_pos_enc = torch.cat((word_pos_out, sent_pos_out), dim=-1)

        # [batch_size, n_blocks, n_tokens, d_model]
        embed_out = word_embed_out + combined_pos_enc
        embed_out = self.enc_embed_dropout(embed_out)

        # [batch_size * n_blocks, n_tokens, d_model]
        embed_out = embed_out.contiguous().view(-1, self.max_para_len, self.embed_size)

        # [batch_size * n_blocks, n_heads, n_tokens, n_tokens]
        src_words_self_attn_bias = src_words_self_attn_bias.contiguous().view(
            -1, self.n_heads, self.max_para_len, self.max_para_len
        )

        # [batch_size * n_blocks, n_tokens, d_model]
        enc_words_out = self.transformer_encoder(embed_out, src_words_self_attn_bias)

        # [batch_size, n_blocks, d_model]
        enc_sents_out = self.graph_encoder(
            enc_words_out, src_words_self_attn_bias, src_sent_self_attn_bias, graph_attn_bias
        )

        enc_words_out = self.enc_layer_norm(enc_words_out)

        # [batch_size, n_blocks, n_tokens, d_model]
        enc_words_out = enc_words_out.contiguous().view(
            -1, self.max_para_num, self.max_para_len, self.embed_size
        )

        # [batch_size, n_blocks, n_tokens, d_model] [batch_size, n_blocks, d_model]
        return enc_words_out, enc_sents_out

    def decode(self, dec_input, enc_words_out, enc_sents_out, state=None):
        tgt_word, tgt_pos, tgt_self_attn_bias, tgt_src_words_attn_bias, tgt_src_sents_attn_bias,\
            graph_attn_bias, tgt_topic, tgt_topic_attn_bias, para_topic, para_topic_attn_bias = dec_input

        # [batch_size, tgt_len, d_model]
        embed_out = self.dec_embed(tgt_word)
        embed_out = embed_out * (self.embed_size ** 0.5)

        # [batch_size, tgt_len, d_model]
        pos_embed_out = self.dec_pos_embed(tgt_pos)

        # [batch_size, tgt_len, d_model]
        embed_out = embed_out + pos_embed_out
        embed_out = self.dec_embed_dropout(embed_out)

        tgt_topic_embed_out = self.tgt_topic_embed(tgt_topic)
        para_topic_embed_out = self.para_topic_embed(para_topic)

        # [batch_size, tgt_len, d_model]
        dec_output = self.graph_decoder(
            embed_out, enc_words_out, enc_sents_out, tgt_self_attn_bias,
            tgt_src_words_attn_bias, tgt_src_sents_attn_bias, graph_attn_bias,
            tgt_topic_embed_out, tgt_topic_attn_bias, para_topic_embed_out, para_topic_attn_bias,
            state=state
        )

        # [batch_size * tgt_len, d_model]
        dec_output = dec_output.contiguous().view(-1, self.embed_size)

        # [batch_size * tgt_len, vocab_size]
        predict = self.generator_fc(dec_output)
        predict = self.generator_log_softmax(predict)

        # [batch_size * tgt_len, vocab_size]
        return predict

    def forward(self, enc_input, dec_input):
        enc_words_out, enc_sents_out = self.encode(enc_input)
        dec_out = self.decode(dec_input, enc_words_out, enc_sents_out)

        return dec_out
