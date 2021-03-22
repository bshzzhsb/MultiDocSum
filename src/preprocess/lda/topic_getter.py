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
        self.model = ProdLDA(args, checkpoint=checkpoint)
        self.model.eval()

    def get_topic(self, src):
        src = self.vectorizer.transform(src).toarray()
        src = torch.tensor(src, dtype=torch.float32, device=self.device)
        probs, _, _, _ = self.model.encode(src)

        topic_words = probs.numpy().argsort()[: -10 - 1: -1]
        return topic_words
