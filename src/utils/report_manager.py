from datetime import datetime

from utils.logger import logger
from utils.statistics import Statistics


def build_report_manager(report_every, tensorboard_log_dir, tensorboard=True):
    if tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard_log_dir + datetime.now().strftime('/%b-%d_%H-%M-%S'))
    else:
        writer = None

    report_manager = ReportManager(report_every, start_time=-1, tensorboard_writer=writer)

    return report_manager


class ReportManager(object):

    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        super(ReportManager, self).__init__()
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time
        self.tensorboard_writer = tensorboard_writer

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(prefix, self.tensorboard_writer, learning_rate, step)

    def report_training(self, step, num_steps, learning_rate, report_stats):
        if self.start_time < 0:
            raise ValueError("ReportManager needs to be started")

        if step % self.report_every == 0:
            report_stats.output(step, num_steps, learning_rate, self.start_time)

            self.log_tensorboard(report_stats, "progress", learning_rate, step)
            report_stats = Statistics()
            self.progress_step += 1

        return report_stats

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())
            self.log_tensorboard(train_stats, 'train', lr, step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())
            self.log_tensorboard(valid_stats, 'valid', lr, step)
