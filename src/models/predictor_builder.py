import torch
import torch.nn
import math
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.logger import logger
from utils.tensor_util import tile
from modules.data_loader import get_num_examples
from utils.cal_rouge import rouge_results_to_str, test_rouge
from utils.beam_search import BeamSearch


def build_predictor(args, tokenizer, symbols, model, device):
    tensorboard_log_dir = args.model_path + '/tensorboard' + '/test'
    writer = SummaryWriter(tensorboard_log_dir)

    translator = Translator(args, model, tokenizer, symbols, device, writer)
    return translator


class Translator(object):

    def __init__(self, args, model, spm, symbols, device, writer=None, n_best=1):
        self.args = args
        self.model = model
        self.vocab = spm
        self.device = device
        self.n_best = n_best
        self.writer = writer

        self.bos_idx = symbols['BOS']
        self.eos_idx = symbols['EOS']
        self.pad_idx = symbols['PAD']
        self.space_idx = symbols['SPACE']

        self.beam_size = self.args.beam_size
        self.batch_size = self.args.batch_size

        self.min_out_len = self.args.min_out_len
        self.max_out_len = self.args.max_out_len
        self.result_path = self.args.result_path

        self.length_penalty = self.args.length_penalty
        self.blocking_trigram = self.args.block_trigram

        self.id2is_full_token = [self.vocab.IdToPiece(token_id).startswith('▁')
                                 for token_id in range(len(self.vocab))]

    def translate(self, test_iter, step):
        logger.info('Start predicting')
        self.model.eval()

        gold_path = self.result_path + '/res.%d.gold' % step
        candi_path = self.result_path + '/res.%d.candidate' % step
        raw_gold_path = self.result_path + '/res.%d.raw_gold' % step
        raw_candi_path = self.result_path + '/res.%d.raw_candidate' % step
        raw_src_path = self.result_path + '/res.%d.raw_src' % step
        gold_file = open(gold_path, 'w', encoding='utf-8')
        candi_file = open(candi_path, 'w', encoding='utf-8')
        raw_gold_file = open(raw_gold_path, 'w', encoding='utf-8')
        raw_candi_file = open(raw_candi_path, 'w', encoding='utf-8')
        raw_src_file = open(raw_src_path, 'w', encoding='utf-8')

        with torch.no_grad():
            total = math.ceil(get_num_examples(self.args.data_path, self.args.mode) / self.batch_size)
            for batch in tqdm(test_iter, total=total):
                self.batch_size = batch.batch_size
                batch_data = self.translate_batch(batch, self.n_best)

                translations = self.from_batch(batch_data)
                for translation in translations:
                    pred, gold, src = translation
                    pred_str = ' '.join(pred).replace('<Q>', ' ').replace(' +', ' ') \
                        .replace('<unk>', 'UNK').replace('\\', '').strip()
                    gold_str = ' '.join(gold).replace('<t>', '').replace('</t>', '') \
                        .replace('<Q>', ' ').replace(' +', ' ').replace('\\', '').strip()

                    gold_str = gold_str.lower()
                    raw_candi_file.write(' '.join(pred).strip() + ' \n')
                    raw_gold_file.write(' '.join(gold).strip() + ' \n')
                    candi_file.write(pred_str + ' \n')
                    gold_file.write(gold_str + ' \n')
                    raw_src_file.write(src.strip() + ' \n')

                raw_candi_file.flush()
                raw_gold_file.flush()
                candi_file.flush()
                gold_file.flush()
                raw_src_file.flush()

        raw_candi_file.close()
        raw_gold_file.close()
        candi_file.close()
        gold_file.close()
        raw_src_file.close()

        if step != -1 and self.args.report_rouge:
            rouges = self._report_rouge(gold_path, candi_path)
            logger.info(rouges)
            logger.info('Rouges at step %d \n %s' % (step, rouge_results_to_str(rouges)))
            if self.writer is not None:
                self.writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def translate_batch(self, batch, n_best=1):
        batch_size = self.batch_size
        beam_size = self.beam_size

        enc_input, dec_input = batch.enc_input, batch.dec_input
        _, _, _, src_words_self_attn_bias, src_sents_self_attn_bias, graph_attn_bias = enc_input
        tgt_topic, tgt_topic_attn_bias, para_topic, para_topic_attn_bias = dec_input[6:]
        tgt_topic = tile(tgt_topic, beam_size, 0)
        tgt_topic_attn_bias = tile(tgt_topic_attn_bias[:, :, 0].unsqueeze(2), beam_size, 0)
        para_topic = tile(para_topic, beam_size, 0)
        para_topic_attn_bias = tile(para_topic_attn_bias, beam_size, 0)

        # [batch_size, max_para_num, n_heads, 1, max_para_len]
        tgt_src_words_attn_bias = src_words_self_attn_bias[:, :, :, 0].unsqueeze(3)
        tgt_src_words_attn_bias.requires_grad = False
        # [batch_size, n_heads, 1, max_para_num]
        tgt_src_sents_attn_bias = src_sents_self_attn_bias[:, :, 0].unsqueeze(2)
        tgt_src_sents_attn_bias.requires_grad = False

        # 拿到 encoder 的输出，并展开 beam_size 维度
        enc_words_output, enc_sents_output = self.model.encode(enc_input)
        enc_words_output = tile(enc_words_output, beam_size, 0)
        enc_sents_output = tile(enc_sents_output, beam_size, 0)

        dec_state = self.model.graph_decoder.init_decoder_state(with_cache=True)

        batch_offset = torch.arange(batch_size, dtype=torch.int64, device=self.device)
        beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.int64, device=self.device)

        alive_seq = torch.full([batch_size * beam_size, 1], self.bos_idx, dtype=torch.int64, device=self.device)

        # 初始化时，第一个 beam 的概率为 1
        topk_log_probs = torch.tensor([0.0] + [float('-inf')] * (beam_size - 1), device=self.device).repeat(batch_size)

        hypotheses = [[] for _ in range(batch_size)]

        results = {
            'predictions': [[] for _ in range(batch_size)],
            'scores': [[] for _ in range(batch_size)],
            'gold_score': [0] * batch_size,
            'batch': batch
        }
        # [batch_size * beam_size, 1]
        pre_src_words_attn_bias = tile(tgt_src_words_attn_bias, beam_size, 0)
        pre_src_sents_attn_bias = tile(tgt_src_sents_attn_bias, beam_size, 0)
        pre_graph_attn_bias = tile(graph_attn_bias, beam_size, 0)

        for step in range(self.max_out_len):
            pre_ids = alive_seq[:, -1].view(-1, 1)
            pre_pos = torch.full_like(pre_ids, step, dtype=torch.int64, device=self.device)
            pre_scores = topk_log_probs

            dec_input = (pre_ids, pre_pos, None, pre_src_words_attn_bias, pre_src_sents_attn_bias,
                         pre_graph_attn_bias, tgt_topic, tgt_topic_attn_bias, para_topic, para_topic_attn_bias)

            # [batch_size * beam_size, vocab_size]
            logits = self.model.decode(dec_input, enc_words_output, enc_sents_output, dec_state)

            if step < self.min_out_len:
                logits[:, self.eos_idx] = -1e20

            # 第一次：从每个 beam 中选取 topk 个候选，相当于每个 batch 有 beam_size * beam_size 个候选
            # [batch_size * beam_size, beam_size]
            topk_vocab_scores, topk_vocab_indices = logits.topk(beam_size, dim=-1)

            if self.blocking_trigram:
                expand_alive_seq = alive_seq.unsqueeze(1).expand(-1, beam_size, -1)
                # [batch_size * beam_size, beam_size, step + 1]
                candi_seqs = torch.cat([expand_alive_seq, topk_vocab_indices.unsqueeze(-1)], dim=-1)
                # [batch_size * beam_size, beam_size]
                mask_block_trigram = self.block_trigram(candi_seqs)
                # [batch_size * beam_size, beam_size]
                pre_scores = topk_log_probs.view(-1).unsqueeze(-1) + mask_block_trigram

            topk_vocab_scores = topk_vocab_scores + pre_scores
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** self.length_penalty
            # [batch_size * beam_size, beam_size]
            cur_scores = topk_vocab_scores / length_penalty

            # 将第一维变成 batch_size，来从每个 batch 的所有 beam 中选择最优的 topk 个
            # [batch_size, beam_size * beam_size]
            cur_scores = cur_scores.view(-1, beam_size * beam_size)
            topk_vocab_indices = topk_vocab_indices.view(-1, beam_size * beam_size)

            # 第二次：从每个 batch 的 beam_size * beam_size 个候选里，选取最好的 beam_size 个
            # [batch_size, beam_size]
            topk_scores, topk_indices = cur_scores.topk(beam_size, dim=-1)

            topk_log_probs = topk_scores * length_penalty

            # 在哪个 beam 里面 [batch_size, beam_size]
            topk_beam_index = topk_indices.div(beam_size).to(torch.int64)

            # 更新 topk_indices
            # [batch_size, beam_size]
            topk_ids = topk_vocab_indices.gather(1, topk_indices)

            # 在 batch * beam 中的序号 [batch_size, beam_size]
            batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)

            # [batch_size, beam_size]
            is_finished = topk_ids.view(-1, beam_size).eq(self.eos_idx)
            if step + 1 == self.max_out_len:
                is_finished.fill_(True)
            # 结束条件：top beam 结束
            end_condition = is_finished[:, 0].eq(True)

            if is_finished.any():
                # [batch_size, beam_size, step + 1]
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(True)
                    # 把完成的 beam 加入结果集
                    finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))

                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results['scores'][b].append(score)
                            results['predictions'][b].append(pred)
                non_finished = (~end_condition).nonzero(as_tuple=False).view(-1)
                if len(non_finished) == 0:
                    break
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))

                # 有 batch 完成时，将不会再对其预测
                select_indices = batch_index.view(-1)
                enc_words_output = enc_words_output.index_select(0, select_indices)
                enc_sents_output = enc_sents_output.index_select(0, select_indices)
                pre_src_words_attn_bias = pre_src_words_attn_bias.index_select(0, select_indices)
                pre_src_sents_attn_bias = pre_src_sents_attn_bias.index_select(0, select_indices)
                tgt_topic = tgt_topic.index_select(0, select_indices)
                tgt_topic_attn_bias = tgt_topic_attn_bias.index_select(0, select_indices)

            dec_state.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))

        return results

    def translate_b(self, batch, n_best=1):
        batch_size, beam_size = self.batch_size, self.beam_size

        enc_input, dec_input = batch.enc_input, batch.dec_input
        _, _, _, src_words_self_attn_bias, src_sents_self_attn_bias, graph_attn_bias = enc_input
        tgt_topic, tgt_topic_attn_bias = dec_input[6:8]
        tgt_topic = tile(tgt_topic, beam_size, 0)
        tgt_topic_attn_bias = tile(tgt_topic_attn_bias[:, :, 0].unsqueeze(2), beam_size, 0)

        # [batch_size, max_para_num, n_heads, 1, max_para_len]
        tgt_src_words_attn_bias = src_words_self_attn_bias[:, :, :, 0].unsqueeze(3)
        tgt_src_words_attn_bias.requires_grad = False
        # [batch_size, n_heads, 1, max_para_num]
        tgt_src_sents_attn_bias = src_sents_self_attn_bias[:, :, 0].unsqueeze(2)
        tgt_src_sents_attn_bias.requires_grad = False

        # 拿到 encoder 的输出，并展开 beam_size 维度
        enc_words_output, enc_sents_output = self.model.encode(enc_input)
        enc_words_output = tile(enc_words_output, beam_size, 0)
        enc_sents_output = tile(enc_sents_output, beam_size, 0)

        dec_state = self.model.graph_decoder.init_decoder_state(with_cache=True)

        alive_seq = torch.full([batch_size * beam_size, 1], self.bos_idx, dtype=torch.int64, device=self.device)
        batch_beam_size, cur_len = alive_seq.size()

        beam_scores = torch.zeros([batch_size, beam_size], dtype=torch.float, device=self.device)
        beam_scores[:, 1:] = -1e20
        beam_scores = beam_scores.view(-1)

        # [batch_size * beam_size, 1]
        pre_src_words_attn_bias = tile(tgt_src_words_attn_bias, beam_size, 0)
        pre_src_sents_attn_bias = tile(tgt_src_sents_attn_bias, beam_size, 0)
        pre_graph_attn_bias = tile(graph_attn_bias, beam_size, 0)

        beam_search = BeamSearch(batch_size, self.max_out_len, beam_size, n_best, self.length_penalty, self.device)

        while cur_len < self.max_out_len:
            pre_ids = alive_seq[:, -1].view(-1, 1)
            pre_pos = torch.full_like(pre_ids, cur_len - 1, dtype=torch.int64, device=self.device)

            dec_input = (pre_ids, pre_pos, None, pre_src_words_attn_bias,
                         pre_src_sents_attn_bias, pre_graph_attn_bias, tgt_topic, tgt_topic_attn_bias)

            # [batch_size * beam_size, vocab_size]
            next_token_scores = self.model.decode(dec_input, enc_words_output, enc_sents_output, dec_state)
            if cur_len < self.min_out_len:
                next_token_scores[:, self.eos_idx] = -1e20
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1).expand_as(next_token_scores)

            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(-1, beam_size * vocab_size)
            # [batch_size, 2 * beam_size]
            next_token_scores, next_tokens = next_token_scores.topk(beam_size * beam_size, dim=-1)
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            if self.blocking_trigram:
                # [batch_size, 2 * beam_size, cur_len]
                expand_alive_seq = alive_seq.view(-1, beam_size, cur_len).gather(1, next_indices.unsqueeze(-1).expand(-1, -1, cur_len))
                # [batch_size, 2 * beam_size, cur_len + 1]
                candi_seqs = torch.cat([expand_alive_seq, next_tokens.unsqueeze(-1)], dim=-1)
                # [batch_size, 2 * beam_size]
                mask_block_trigram = self.block_trigram(candi_seqs)
                # [batch_size, 2 * beam_size]
                next_token_scores = next_token_scores + mask_block_trigram

            beam_outputs = beam_search.process(
                alive_seq, next_token_scores, next_tokens, next_indices,
                self.pad_idx, self.eos_idx
            )
            beam_scores, beam_next_tokens, beam_indices = \
                beam_outputs['next_beam_scores'], beam_outputs['next_beam_tokens'], beam_outputs['next_beam_indices']

            alive_seq = torch.cat([alive_seq[beam_indices, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len += 1
            dec_state.map_batch_fn(lambda state, dim: state.index_select(dim, beam_indices))

            if beam_search.is_done:
                break

        sequence_outputs = beam_search.finalize(alive_seq, beam_scores, self.pad_idx, self.eos_idx)
        results = {
            'predictions': sequence_outputs['sequences'],
            'scores': sequence_outputs['sequence_scores'],
            'gold_score': [0] * batch_size,
            'batch': batch
        }
        return results

    def from_batch(self, translation_batch):
        batch = translation_batch['batch']
        assert len(translation_batch['gold_score']) == len(translation_batch['predictions'])
        batch_size = self.batch_size

        preds, pred_score, gold_score, tgt_str, src = list(zip(
            *list(zip(
                translation_batch['predictions'],
                translation_batch['scores'],
                translation_batch['gold_score'],
                batch.tgt_str,
                batch.enc_input[0],
            ))
        ))

        translations = []
        for b in range(batch_size):
            pred_sent = sum([self._build_target_tokens(preds[b][n]) for n in range(self.n_best)], [])
            gold_sent = tgt_str[b].split()

            raw_src = '<PARA>'.join([self.vocab.DecodeIds([int(w) for w in t]) for t in src[b]])
            translation = (pred_sent, gold_sent, raw_src)
            translations.append(translation)

        return translations

    def _build_target_tokens(self, pred):
        tokens = []
        for tok in pred:
            tok = int(tok)
            if tok == self.eos_idx:
                break
            tokens.append(tok)
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def _report_rouge(self, gold_path, candi_path):
        logger.info('Calculating Rouge')
        candidates = open(candi_path, encoding='utf-8')
        references = open(gold_path, encoding='utf-8')
        result_dict = test_rouge(candidates, references, 8)
        return result_dict

    def block_trigram(self, candi_seq):
        """
        :param candi_seq: [batch_size * beam_size, beam_size, len]
        """
        def _sub_token_id2full_tokens(sub_token_indices):
            full_tokens = []
            pre_is_space = False
            for sub_token_id in sub_token_indices.cpu().numpy().tolist():
                is_full_token = self.id2is_full_token[sub_token_id]
                if sub_token_id == self.space_idx:
                    pre_is_space = True
                    continue
                if is_full_token or not full_tokens or pre_is_space:
                    full_tokens.append([sub_token_id])
                else:
                    pre_full_token = full_tokens[-1]
                    pre_full_token.append(sub_token_id)
                pre_is_space = False

            full_tokens = ['-'.join(map(str, full_token)) for full_token in full_tokens]
            return full_tokens

        def _make_trigram_str(trigram_tokens):
            return '_'.join(trigram_tokens)

        delta_list = []
        for beam in candi_seq:
            delta_scores = []
            for seq in beam:
                sub_token_ids = seq.view(-1)
                tokens = _sub_token_id2full_tokens(sub_token_ids)
                if len(tokens) <= 3:
                    delta_scores.append(0)
                    continue
                trigrams = [_make_trigram_str(tokens[end - 3: end]) for end in range(3, len(tokens))]
                trigrams_set = set(trigrams)
                last_trigram = _make_trigram_str(tokens[-3:])
                if last_trigram in trigrams_set:
                    # 重复的 trigram
                    delta_scores.append(-1e20)
                else:
                    delta_scores.append(0)
            delta_list.append(delta_scores)
        # [batch_size * beam_size, beam_size]
        return torch.tensor(delta_list, dtype=torch.float, device=self.device)
