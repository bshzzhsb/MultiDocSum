import torch
import torch.nn
from tensorboardX import SummaryWriter

from utils.logging import logger
from utils.tensor_util import tile
from utils.cal_rouge import rouge_results_to_str, test_rouge


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

        self.beam_size = self.args.beam_size
        self.batch_size = self.args.batch_size

        self.min_out_len = self.args.min_out_len
        self.max_out_len = self.args.max_out_len
        self.result_path = self.args.result_path

        self.length_penalty = self.args.length_penalty
        self.blocking_trigram = self.args.block_trigram

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
            for batch in test_iter:
                batch_data = self.translate_batch(batch, self.n_best)

                translations = self.from_batch(batch_data)
                for translation in translations:
                    pred, gold, src = translation
                    pred_str = ' '.join(pred).replace('<Q>', ' ').replace(r' +', ' ') \
                                  .replace('<unk>', 'UNK').strip()
                    gold_str = ' '.join(gold).replace('<t>', '').replace('</t>', '') \
                                  .replace('<Q>', ' ').replace(r' +', ' ').strip()

                    gold_str = gold_str.lower()
                    raw_candi_file.write(' '.join(pred).strip() + '\n')
                    raw_gold_file.write(' '.join(gold).strip() + '\n')
                    candi_file.write(pred_str + '\n')
                    gold_file.write(gold_str + '\n')
                    raw_src_file.write(src.strip() + '\n')
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
            logger.info('Rouges at step %d \n %s' % (step, rouge_results_to_str(rouges)))
            if self.writer is not None:
                self.writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def translate_batch(self, batch, n_best=1):
        batch_size = self.batch_size
        beam_size = self.beam_size

        enc_input = batch.enc_input
        _, _, _, src_words_self_attn_bias, src_sents_self_attn_bias, graph_attn_bias = enc_input

        # [batch_size, max_para_num, n_heads, 1, max_para_len]
        tgt_src_words_attn_bias = src_words_self_attn_bias[:, :, :, 0].unsqueeze(3)
        tgt_src_words_attn_bias.requires_grad = False
        # [batch_size, n_heads, 1, max_para_num]
        tgt_src_sents_attn_bias = src_sents_self_attn_bias[:, :, 0].unsqueeze(2)
        tgt_src_sents_attn_bias.requires_grad = False

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

            dec_input = (pre_ids, pre_pos, None, pre_src_words_attn_bias,
                         pre_src_sents_attn_bias, pre_graph_attn_bias)

            logits = self.model.decode(dec_input, enc_words_output, enc_sents_output, dec_state)
            vocab_size = logits.size(-1)

            if step < self.min_out_len:
                logits[:, self.eos_idx] = -1e20

            logits += topk_log_probs.view(-1).unsqueeze(1)

            length_penalty = ((5.0 + (step + 1)) / 6.0) ** self.length_penalty

            cur_scores = logits / length_penalty

            if self.blocking_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        words = list(map(lambda w: self.vocab.IdToPiece(int(w)), alive_seq[i]))
                        if len(words) <= 3:
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = trigrams[-1]
                        if trigram in trigrams[:-1]:
                            cur_scores[i] = -1e20

            cur_scores = cur_scores.reshape(-1, beam_size * vocab_size)
            # [batch_size, beam_size]
            topk_scores, topk_ids = cur_scores.topk(beam_size, dim=-1)

            topk_log_probs = topk_scores * length_penalty

            topk_beam_index = topk_ids.div(vocab_size).to(torch.int64)
            topk_ids = topk_ids.fmod(vocab_size)

            batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)

            # [batch_size, beam_size]
            is_finished = topk_ids.eq(self.eos_idx)
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
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))

                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results['scores'][b].append(score)
                            results['predictions'][b].append(pred)
                non_finished = (~end_condition).nonzero().view(-1)
                if len(non_finished) == 0:
                    break
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))

            select_indices = batch_index.view(-1)
            enc_words_output = enc_words_output.index_select(0, select_indices)
            enc_sents_output = enc_sents_output.index_select(0, select_indices)
            pre_src_words_attn_bias = pre_src_words_attn_bias.index_select(0, select_indices)
            pre_src_sents_attn_bias = pre_src_sents_attn_bias.index_select(0, select_indices)

            dec_state.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))

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
        result_dict = test_rouge(candidates, references, 1)
        return result_dict
