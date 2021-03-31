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

    def get_topic(self, src, num_top_topic, num_top_word):
        self.model.eval()
        with torch.no_grad():
            src = self.vectorizer.transform(src).toarray()
            src = torch.tensor(src, dtype=torch.float32, device=self.device)
            probs, _, _, _ = self.model.encode(src)
            vocab_probs = self.model.decode(probs)

            top_n_words = vocab_probs[0].numpy().argsort()[: -num_top_word - 1: -1]
            top_n_words_probs = vocab_probs[0].numpy()[top_n_words]

            top_n_topics = probs[0].numpy().argsort()[: -num_top_topic - 1: -1]
            top_n_topics_probs = probs[0].numpy()[top_n_topics]
        return top_n_topics, top_n_topics_probs, top_n_words, top_n_words_probs
