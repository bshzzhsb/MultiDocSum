import torch
import torch.nn
from sklearn.feature_extraction.text import CountVectorizer

from utils.logger import logger
from preprocess.lda import ProdLDA


class TopicModel(object):

    def __init__(self, args, vocab, device, checkpoint):
        self.args = args
        self.device = device
        self.vocab = vocab

        self.vectorizer = CountVectorizer(vocabulary=self.vocab)

        assert checkpoint is not None

        logger.info('Loading checkpoint from %s' % checkpoint)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        args.num_topics = checkpoint['num_topics']
        self.model = ProdLDA(args, checkpoint=checkpoint)

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
