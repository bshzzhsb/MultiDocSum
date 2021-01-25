import torch
import torch.nn as nn
import json

from modules.encoder import TransformerEncoder, GraphEncoder
from modules.decoder import GraphDecoder
from modules.neural_modules import PrePostProcessLayer
from utils.initializer import truncated_normal


class GraphSumConfig(object):

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing model config path '%s'" % config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        for arg, value in sorted(self._config_dict.items()):
            print('%s: %s' % (arg, value))
        print('--------------------------------------------------')


class GraphSum(nn.Module):

    def __init__(self, args, config, padding_idx, bos_idx, eos_idx, tokenizer):
        super(GraphSum, self).__init__()
        self.args = args
        self.embed_size = config['hidden_size']
        self.enc_word_layers = config['encoder_word_layers']
        self.enc_graph_layers = config['encoder_graph_layers']
        self.dec_graph_layers = config['decoder_graph_layers']
        self.n_heads = config['num_attention_heads']
        self.max_pos_seq_len = config['max_position_embeddings']
        self.hidden_act = config['hidden_activation']
        self.pre_post_process_dropout = config['hidden_dropout_probability']
        self.attn_dropout = config['attention_dropout_probability']
        self.pre_process_cmd = config['pre_process_command']
        self.post_process_cmd = config['post_process_command']
        self.word_embed_name = config['word_embedding_name']
        self.enc_word_pos_embed_name = config['encoder_word_position_embedding_name']
        self.enc_sent_pos_embed_name = config['encoder_sentence_position_embedding_name']
        self.dec_word_pos_embed_name = config['decoder_word_position_embedding_name']

        self.param_initializer = truncated_normal(std=config['initializer_range'])

        self.label_smooth_eps = args.label_smooth_eps
        self.padding_idx = padding_idx
        self.weight_sharing = args.weights_sharing
        self.beam_size = args.beam_size
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.tokenizer = tokenizer
        self.voc_size = len(tokenizer)
        self.max_param_len = args.max_param_len
        self.max_tgt_len = args.max_tgt_len
        self.len_penalty = args.len_penalty
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.block_trigram = args.block_trigram
        self.pos_win = args.pos_win

        self.enc_word_embed = nn.Embedding(self.voc_size, self.args.embed_size, padding_idx)
        self.enc_word_pos_embed = nn.Embedding(self.args.max_pos_seq_len, self.args.embed_size // 2)
        self.enc_sent_pos_embed = nn.Embedding(self.args.max_pos_seq_len, self.args.embed_size // 2)
        self.enc_embed_dropout = nn.Dropout(self.args.pre_post_process_dropout)
        self.dec_embed = nn.Embedding(self.args.voc_size, self.args.embed_size, padding_idx)
        self.dec_pos_embed = nn.Embedding(self.args.max_pos_seq_len, self.args.embed_size)
        self.dec_embed_dropout = nn.Dropout(self.args.pre_post_process_dropout)

        if self.weight_sharing:
            self.dec_embed.weight = self.enc_word_embed.weight

        self.transformer_encoder = TransformerEncoder(
            n_layers=self.args.enc_word_layers,
            with_post_process=False,
            n_heads=self.args.n_heads,
            d_k=self.args.embed_size // self.args.n_heads,
            d_v=self.args.embed_size // self.args.n_heads,
            d_model=self.args.embed_size,
            d_inner_hidden=self.args.embed_size * 4,
            pre_post_process_dropout=self.args.pre_post_process_dropout,
            attn_dropout=self.args.attn_dropout,
            relu_dropout=self.args.pre_post_process_dropout,
            hidden_act=self.args.hidden_act,
            pre_process_cmd=self.args.pre_process_cmd,
            post_process_cmd=self.args.post_process_cmd
        )
        self.graph_encoder = GraphEncoder(
            n_graph_layers=self.args.enc_graph_layers,
            n_heads=self.args.n_heads,
            n_blocks=self.args.max_param_num,
            d_model=self.args.embed_size,
            d_k=self.args.embed_size / self.args.n_heads,
            d_v=self.args.embed_size / self.args.n_heads,
            d_inner_hidden=self.args.embed_size * 4,
            pos_win=self.args.pos_win,
            pre_post_process_dropout=self.args.pre_post_process_dropout,
            attn_dropout=self.args.attn_dropout,
            relu_dropout=self.args.pre_post_process_dropout,
            hidden_act=self.args.hidden_act,
            pre_process_cmd=self.args.pre_process_cmd,
            post_process_cmd=self.args.post_process_cmd
        )

        self.pre_process_layer = PrePostProcessLayer(
            self.args.d_model, self.args.pre_process_cmd, self.args.pre_post_process_dropout
        )

        self.graph_decoder = GraphDecoder(
            n_layers=self.args.dec_n_layers,
            n_heads=self.args.n_heads,
            d_model=self.args.embed_size,
            d_k=self.args.embed_size // self.args.n_heads,
            d_v=self.args.embed_size // self.args.n_heads,
            d_inner_hidden=self.embed_size * 4,
            pos_win=self.pos_win,
            pre_post_process_dropout=self.pre_post_process_dropout,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.pre_post_process_dropout,
            hidden_act=self.hidden_act,
            pre_process_cmd=self.pre_process_cmd,
            post_process_cmd=self.post_process_cmd
        )

        self.fc = nn.Linear(self.embed_size, self.voc_size)

        self.loss = nn.CrossEntropyLoss()

    def encode(self, enc_input):
        src_word, src_word_pos, src_sent_pos, src_words_self_attn_bias, \
            src_sent_self_attn_bias, graph_attn_bias = enc_input

        # [batch_size, max_n_blocks, max_n_tokens, embed_dim]
        word_embed_out = self.enc_word_embed(src_word)
        word_embed_out = word_embed_out * (self.args.embed_size ** 0.5)

        # [batch_size, max_n_blocks, max_n_tokens, embed_dim / 2]
        word_pos_out = self.enc_word_pos_embed(src_word_pos)
        word_pos_out.requires_grad_(requires_grad=False)

        # [batch_size, max_n_blocks, embed_dim / 2]
        sent_pos_out = self.enc_sent_pos_embed(src_sent_pos)
        sent_pos_out.requires_grad_(requires_grad=False)

        # [batch_size, max_n_blocks, max_n_tokens, embed_dim / 2]
        sent_pos_out = torch.unsqueeze(sent_pos_out, 2).expand(-1, -1, self.args.max_param_len, -1)

        # [batch_size, n_blocks, n_tokens, embed_dim]
        combined_pos_enc = torch.cat((word_pos_out, sent_pos_out), dim=-1)

        # [batch_size, n_blocks, n_tokens, embed_dim]
        embed_out = word_embed_out + combined_pos_enc
        embed_out = self.enc_embed_dropout(embed_out)

        embed_out = embed_out.contiguous().view(-1, self.args.max_param_len, self.args.embed_size)

        src_words_self_attn_bias = src_words_self_attn_bias.contiguous().view(
            -1, self.args.n_heads, self.args.max_param_len, self.args.max_param_len
        )

        enc_words_out = self.transformer_encoder(embed_out, src_words_self_attn_bias)

        enc_sents_out = self.graph_encoder(
            enc_words_out, src_words_self_attn_bias, src_sent_self_attn_bias, graph_attn_bias
        )

        enc_words_out = self.pre_process_layer(None, enc_words_out)

        enc_words_out = enc_words_out.contiguous().view(
            -1, self.args.max_param_num, self.args.max_param_len, self.args.embed_size
        )

        return enc_words_out, enc_sents_out

    def decode(self, dec_input, enc_words_out, enc_sents_out):
        trg_word, trg_pos, trg_self_attn_bias, trg_src_words_attn_bias, \
        trg_src_sents_attn_bias, graph_attn_bias = dec_input

        embed_out = self.dec_embed(trg_word)
        embed_out = embed_out * (self.args.embed_size ** 0.5)

        pos_embed_out = self.dec_pos_embed(trg_pos)
        pos_embed_out.requires_grad_(requires_grad=False)

        embed_out = embed_out + pos_embed_out
        embed_out = self.dec_embed_dropout(embed_out)

        dec_output = self.graph_decoder(
            embed_out, enc_words_out, enc_sents_out,
            trg_self_attn_bias, trg_src_words_attn_bias, trg_src_sents_attn_bias, graph_attn_bias
        )

        dec_output = dec_output.contiguous().view(-1, self.embed_size)

        if self.weight_sharing:
            out = torch.matmul(dec_output, self.enc_word_embed.weight)
            bias = nn.Parameter(torch.zeros(self.voc_size, dtype=torch.float32), requires_grad=True)
            predict = out + bias
        else:
            predict = self.fc(dec_output)

        return predict

    def forward(self, enc_input, dec_input, tgt_label, label_weights):
        enc_words_out, enc_sents_out = self.encode(enc_input)
        dec_out = self.decode(dec_input, enc_words_out, enc_sents_out)

        predict_token_idx = torch.argmax(dec_out, dim=-1)
        correct_token_idx = (tgt_label == (predict_token_idx.view(-1, 1))).to(torch.float32)
        weighted_correct = torch.mul(correct_token_idx, label_weights)
        sum_correct = weighted_correct.sum(dim=0)
        sum_correct.requires_grad_(requires_grad=False)

        cost = self.loss(dec_out, tgt_label)

        weighted_cost = torch.mul(cost, label_weights)
        sum_cost = weighted_cost.sum(dim=0)
        token_num = label_weights.sum(dim=0)
        token_num.requires_grad_(requires_grad=False)
        avg_cost = sum_cost / token_num

        graph_vars = {
            "loss": avg_cost,
            "sum_correct": sum_correct,
            "token_num": token_num
        }

        return graph_vars
