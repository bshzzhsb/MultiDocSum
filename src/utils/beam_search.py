import torch


class BeamHypotheses:

    def __init__(self, beam_size, max_length, length_penalty):
        self.max_length = max_length - 1  # 忽略<BOS>
        self.length_penalty = length_penalty
        self.beam_size = beam_size
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        length_penalty = ((5.0 + hyp.size(-1)) / 6.0) ** self.length_penalty
        score = sum_logprobs / length_penalty
        if len(self) < self.beam_size or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.beam_size:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        if len(self) < self.beam_size:
            return False
        else:
            cur_score = best_sum_logprobs / (((5.0 + cur_len) / 6.0) ** self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret


class BeamSearch:

    def __init__(self, batch_size, max_length, beam_size, n_best, length_penalty, device):
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_best = n_best
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.device = device

        self._beam_hyps = [
            BeamHypotheses(
                beam_size=beam_size, max_length=max_length, length_penalty=length_penalty
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(self, input_ids, next_scores, next_tokens, next_indices, pad_idx, eos_idx):
        """
        :param next_indices: 选取的 beam 在 beam_size 中的索引
        """
        batch_size, beam_size, device = self.batch_size, self.beam_size, self.device

        cur_len = input_ids.size(-1)
        next_beam_scores = torch.zeros([batch_size, beam_size], dtype=torch.float, device=device)
        next_beam_tokens = torch.full_like(next_beam_scores, 0, dtype=torch.int64, device=device)
        next_beam_indices = torch.full_like(next_beam_scores, 0, dtype=torch.int64, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_idx
                next_beam_indices[batch_idx, :] = 0
                continue

            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * beam_size + next_index
                if next_token.item() == eos_idx:
                    # 不属于 topk 个 beam 则不添加
                    if beam_token_rank >= beam_size:
                        continue
                    # 将完成的句子添加进集合
                    beam_hyp.add(input_ids[batch_beam_idx].clone(), next_score.item())
                else:
                    # 添加下一个单词
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # 当 beam 满了时，退出循环
                if beam_idx == beam_size:
                    break

            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len)

        return {
            'next_beam_scores': next_beam_scores.view(-1),
            'next_beam_tokens': next_beam_tokens.view(-1),
            'next_beam_indices': next_beam_indices.view(-1)
        }

    def finalize(self, input_ids, final_beam_scores, pad_idx, eos_idx):
        batch_size, beam_size, n_best, device = self.batch_size, self.beam_size, self.n_best, self.device

        # 将所有未完成的句子添加进集合
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue
            for beam_id in range(beam_size):
                batch_beam_idx = batch_idx * beam_size + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        sent_lengths = torch.empty(batch_size * n_best, dtype=torch.int64, device=device)
        best = []
        best_scores = torch.zeros(batch_size * n_best, dtype=torch.float, device=device)

        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(n_best):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[n_best * i + j] = len(best_hyp) - 1

                best.append(best_hyp[1:])
                best_scores[n_best * i + j] = best_score

        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded = torch.empty([batch_size * n_best, sent_max_len], dtype=torch.int64, device=device)

        if sent_lengths.min().item() != sent_lengths.max().item():
            decoded.fill_(pad_idx)

        for i, hypo in enumerate(best):
            decoded[i, :sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_idx

        decoded = decoded.view(batch_size, n_best, sent_max_len)
        return {'sequences': decoded, 'sequence_scores': best_scores}
