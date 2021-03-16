import math
import sys
import time

from utils.logger import logger


class Statistics(object):

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat, update_n_src_words=False):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """计算交叉熵"""
        return self.loss / self.n_words

    def ppl(self):
        """计算困惑度 perplexity"""
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        t = self.elapsed_time()
        logger.info(
            ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; "
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step, num_steps, self.accuracy(), self.ppl(), self.xent(),
               learning_rate, self.n_src_words / (t + 1e-5), self.n_words / (t + 1e-5),
               time.time() - start)
        )
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
