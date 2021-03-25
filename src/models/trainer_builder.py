import os
import torch

from modules.loss import build_loss_compute
from utils.logger import logger
from utils.statistics import Statistics
from utils.report_manager import build_report_manager


def build_trainer(args, device, model, symbols, vocab_size, optim, get_test_iter):
    train_loss = build_loss_compute(symbols, vocab_size, device, train=True, label_smoothing=args.label_smoothing)
    valid_loss = build_loss_compute(symbols, vocab_size, device, train=False)

    shard_size = args.max_generator_batches

    tensorboard_log_dir = args.model_path + '/tensorboard'
    report_manager = build_report_manager(args.report_every, tensorboard_log_dir)
    trainer = Trainer(args, model, optim, shard_size, train_loss, valid_loss, get_test_iter, report_manager)

    n_params = sum([p.nelement() for p in model.parameters()])
    enc, dec = 0, 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('number of parameters: %d' % n_params)

    return trainer


class Trainer(object):

    def __init__(self, args, model, optim, shard_size, train_loss, valid_loss,
                 get_test_iter=None, report_manager=None):
        self.args = args
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.shard_size = shard_size
        self.report_manager = report_manager
        self.get_test_iter = get_test_iter

    def train(self, train_iter_fct, train_steps):
        logger.info('Start training...')

        step = self.optim._step + 1
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            for i, batch in enumerate(train_iter):
                self.model.train()
                num_tokens = batch.tgt_label.ne(self.train_loss.padding_idx).sum()
                normalization = num_tokens.item()
                self._gradient_accumulation(batch, normalization, total_stats, report_stats)

                report_stats = self._report_training(step, train_steps, self.optim.learning_rate, report_stats)

                if step % self.args.save_checkpoint_steps == 0:
                    self._save(step)

                if step % self.args.val_steps == 0:
                    valid_iter = self.get_test_iter()
                    if self.args.do_val and valid_iter:
                        self.validate(step, valid_iter)

                step += 1
                if step > train_steps:
                    break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, step, valid_iter):
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                enc_input, dec_input, tgt_label, label_weight = \
                    batch.enc_input, batch.dec_input, batch.tgt_label, batch.label_weight
                output = self.model(enc_input, dec_input)
                batch_stats = self.valid_loss.monolithic_compute_loss(tgt_label, output)
                stats.update(batch_stats)
            self._report_step(self.optim.learning_rate, step, valid_stats=stats)
            return stats

    def _gradient_accumulation(self, batch, normalization, total_stats, report_stats):
        enc_input, dec_input, tgt_label, label_weight = \
            batch.enc_input, batch.dec_input, batch.tgt_label, batch.label_weight

        self.model.zero_grad()

        output = self.model(enc_input, dec_input)

        batch_stats = self.train_loss.sharded_compute_loss(
            tgt_label, output, self.shard_size, normalization
        )

        report_stats.n_src_words += enc_input[0].nelement()

        total_stats.update(batch_stats)
        report_stats.update(batch_stats)

        self.optim.step()

    def _save(self, step):
        checkpoint = {
            'step': step,
            'model': self.model.state_dict(),
            'opt': self.args,
            'optim': self.optim.optimizer.state_dict()
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _report_training(self, step, num_steps, learning_rate, report_stats):
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats
            )

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        if self.report_manager is not None:
            self.report_manager.report_step(
                lr, step, train_stats=train_stats, valid_stats=valid_stats
            )
