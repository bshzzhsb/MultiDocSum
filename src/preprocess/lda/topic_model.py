import torch
import torch.nn
from sklearn.feature_extraction.text import CountVectorizer

from utils.logger import logger
from preprocess.lda import ProdLDA


class TopicModel(object):

    def __init__(self, vocab, device, checkpoint):
        self.device = device
        self.vocab = vocab

        self.vectorizer = CountVectorizer(vocabulary=self.vocab)

        assert checkpoint is not None

        logger.info('Loading checkpoint from %s' % checkpoint)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        args = checkpoint['opt']
        num_topics, enc1_units, enc2_units, vocab_size, variance, dropout, init_mult = \
            args.num_topics, args.enc1_units, args.enc2_units, args.vocab_size, \
            args.variance, args.dropout, args.init_mult
        self.model = ProdLDA(num_topics, args.enc1_units, args.enc2_units, vocab_size, variance, dropout,
                             device, init_mult, checkpoint=checkpoint)

    def get_topic_words(self, src, n_topic_words):
        self.model.eval()
        with torch.no_grad():
            src = self.vectorizer.transform(src).toarray()
            src = torch.tensor(src, dtype=torch.float32, device=self.device)
            probs, _, _, _ = self.model.encode(src)
            vocab_probs = self.model.decode(probs)

            topk_scores, topk_indices = vocab_probs.topk(n_topic_words, dim=-1)

            top_n_words = vocab_probs[0].cpu().numpy().argsort()[: -n_topic_words - 1: -1]
            top_n_words_probs = vocab_probs[0].cpu().numpy()[top_n_words]

        return top_n_words, top_n_words_probs

    def get_srcs_topic_words(self, srcs, n_topic_words):
        self.model.eval()
        with torch.no_grad():
            srcs = self.vectorizer.transform(srcs).toarray()
            srcs = torch.tensor(srcs, dtype=torch.float32, device=self.device)
            probs, _, _, _ = self.model.encode(srcs)
            vocab_probs = self.model.decode(probs)

            topk_scores, topk_indices = vocab_probs.sum(0).topk(n_topic_words, dim=-1)
            topk_words = [self.vocab[index] for index in topk_indices]

        return topk_scores, topk_indices, topk_words
