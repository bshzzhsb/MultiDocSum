import torch
import torch.nn as nn
from torch.nn.init import normal_

from graph_sum.neural_modules.encoder import TransformerEncoder, GraphEncoder
from graph_sum.neural_modules.decoder import GraphDecoder
from graph_sum.neural_modules.neural_modules import PreProcessLayer, PositionalEncoding


class GraphSum(nn.Module):

    def __init__(self, args, padding_idx, bos_idx, eos_idx, tokenizer, device,
                 checkpoint=None):
        super(GraphSum, self).__init__()
        self.args = args
        self.embed_size = args.hidden_size
        self.enc_word_layers = args.enc_word_layers
        self.enc_graph_layers = args.enc_graph_layers
        self.dec_graph_layers = args.dec_graph_layers
        self.n_heads = args.n_heads
        self.max_pos_seq_len = args.max_pos_embed
        self.hidden_act = args.hidden_act
        self.pre_post_process_dropout = args.hidden_dropout_prob
        self.attn_dropout = args.attn_dropout_prob
        self.pre_process_cmd = args.pre_process_cmd
        self.post_process_cmd = args.post_process_cmd
        self.initializer_std = args.initializer_range
        self.sc_pos_embed = args.sc_pos_embed

        self.d_model = self.embed_size
        self.padding_idx = padding_idx
        self.weight_sharing = args.weight_sharing
        self.beam_size = args.beam_size
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.max_para_num = args.max_para_num
        self.max_para_len = args.max_para_len
        self.max_tgt_len = args.max_tgt_len
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.block_trigram = args.block_trigram
        self.pos_win = args.pos_win

        self.encoder_word_embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx)
        self.encoder_embedding_dropout = nn.Dropout(self.pre_post_process_dropout)
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx)
        self.decoder_embedding_dropout = nn.Dropout(self.pre_post_process_dropout)
        if self.sc_pos_embed:
            self.encoder_word_pos_embedding = PositionalEncoding(self.embed_size // 2)
            self.encoder_sent_pos_embedding = PositionalEncoding(self.embed_size // 2)
            self.decoder_pos_embedding = PositionalEncoding(self.embed_size)
        else:
            self.encoder_word_pos_embedding = nn.Embedding(self.max_pos_seq_len, self.embed_size // 2)
            self.encoder_word_pos_embedding.weight.requires_grad = False
            self.encoder_sent_pos_embedding = nn.Embedding(self.max_pos_seq_len, self.embed_size // 2)
            self.encoder_sent_pos_embedding.weight.requires_grad = False
            self.decoder_pos_embedding = nn.Embedding(self.max_pos_seq_len, self.embed_size)
            self.decoder_pos_embedding.weight.requires_grad = False

        if self.weight_sharing:
            self.decoder_embedding.weight = self.encoder_word_embedding.weight

        self.transformer_encoder = TransformerEncoder(
            n_layers=self.enc_word_layers,
            with_post_process=False,
            n_heads=self.n_heads,
            d_k=self.embed_size // self.n_heads,
            d_v=self.embed_size // self.n_heads,
            d_model=self.embed_size,
            d_inner_hidden=self.embed_size * 4,
            pre_post_process_dropout=self.pre_post_process_dropout,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.pre_post_process_dropout,
            hidden_act=self.hidden_act,
            pre_process_cmd=self.pre_process_cmd,
            post_process_cmd=self.post_process_cmd
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
            pre_post_process_dropout=self.pre_post_process_dropout,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.pre_post_process_dropout,
            hidden_act=self.hidden_act,
            pre_process_cmd=self.pre_process_cmd,
            post_process_cmd=self.post_process_cmd
        )

        self.post_encoder = PreProcessLayer(
            self.d_model, self.pre_process_cmd, self.pre_post_process_dropout
        )

        self.graph_decoder = GraphDecoder(
            n_layers=self.dec_graph_layers,
            n_heads=self.n_heads,
            d_model=self.embed_size,
            d_k=self.embed_size // self.n_heads,
            d_v=self.embed_size // self.n_heads,
            d_inner_hidden=self.embed_size * 4,
            pos_win=self.pos_win,
            pre_post_process_dropout=self.pre_post_process_dropout,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.pre_post_process_dropout,
            hidden_act=self.hidden_act,
            pre_process_cmd=self.pre_process_cmd,
            post_process_cmd=self.post_process_cmd,
            device=device
        )

        self.generator_fc = nn.Linear(self.embed_size, self.vocab_size)
        if self.weight_sharing:
            self.generator_fc.weight = self.decoder_embedding.weight
            self.generator_fc.bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float32), requires_grad=True)

        self.generator_log_softmax = nn.LogSoftmax(dim=-1)

        if checkpoint is not None:
            keys = list(checkpoint['model'].keys())
            for k in keys:
                if 'a_2' in k:
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del checkpoint['model'][k]
                if 'b_2' in k:
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del checkpoint['model'][k]

            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for name, param in self.named_parameters():
                if 'pos' not in name and param.dim() > 1:
                    normal_(param, mean=0, std=self.initializer_std)

        self.to(device)

    def encode(self, enc_input):
        src_word, src_word_pos, src_sent_pos, src_words_self_attn_bias, \
            src_sent_self_attn_bias, graph_attn_bias = enc_input

        # [batch_size, max_n_blocks, max_n_tokens, embed_dim]
        word_embed_out = self.encoder_word_embedding(src_word)
        word_embed_out = word_embed_out * (self.embed_size ** 0.5)

        # [batch_size, max_n_blocks, max_n_tokens, embed_dim / 2]
        word_pos_out = self.encoder_word_pos_embedding(src_word_pos)

        # [batch_size, max_n_blocks, embed_dim / 2]
        sent_pos_out = self.encoder_sent_pos_embedding(src_sent_pos)

        # [batch_size, max_n_blocks, max_n_tokens, embed_dim / 2]
        sent_pos_out = torch.unsqueeze(sent_pos_out, 2).expand(-1, -1, self.max_para_len, -1)

        # [batch_size, n_blocks, n_tokens, embed_dim]
        combined_pos_enc = torch.cat((word_pos_out, sent_pos_out), dim=-1)

        # [batch_size, n_blocks, n_tokens, embed_dim]
        embed_out = word_embed_out + combined_pos_enc
        embed_out = self.encoder_embedding_dropout(embed_out)

        embed_out = embed_out.contiguous().view(-1, self.max_para_len, self.embed_size)

        src_words_self_attn_bias = src_words_self_attn_bias.contiguous().view(
            -1, self.n_heads, self.max_para_len, self.max_para_len
        )

        enc_words_out = self.transformer_encoder(embed_out, src_words_self_attn_bias)

        enc_sents_out = self.graph_encoder(
            enc_words_out, src_words_self_attn_bias, src_sent_self_attn_bias, graph_attn_bias
        )

        enc_words_out = self.post_encoder(enc_words_out)

        enc_words_out = enc_words_out.contiguous().view(
            -1, self.max_para_num, self.max_para_len, self.embed_size
        )

        return enc_words_out, enc_sents_out

    def decode(self, dec_input, enc_words_out, enc_sents_out, state=None):
        tgt_word, tgt_pos, tgt_self_attn_bias, tgt_src_words_attn_bias, \
            tgt_src_sents_attn_bias, graph_attn_bias = dec_input

        embed_out = self.decoder_embedding(tgt_word)
        embed_out = embed_out * (self.embed_size ** 0.5)

        pos_embed_out = self.decoder_pos_embedding(tgt_pos)

        embed_out = embed_out + pos_embed_out
        embed_out = self.decoder_embedding_dropout(embed_out)

        dec_output = self.graph_decoder(
            embed_out, enc_words_out, enc_sents_out, tgt_self_attn_bias,
            tgt_src_words_attn_bias, tgt_src_sents_attn_bias, graph_attn_bias,
            state=state
        )

        dec_output = dec_output.contiguous().view(-1, self.embed_size)

        predict = self.generator_fc(dec_output)

        predict = self.generator_log_softmax(predict)
        return predict

    def forward(self, enc_input, dec_input):
        enc_words_out, enc_sents_out = self.encode(enc_input)
        dec_out = self.decode(dec_input, enc_words_out, enc_sents_out)

        return dec_out
