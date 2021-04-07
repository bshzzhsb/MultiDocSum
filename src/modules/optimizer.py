import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def build_optim(args, model: nn.Module, checkpoint):
    optimizer = Optimizer(
        method=args.optimizer, learning_rate=args.lr, max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
        model_size=args.hidden_size,
        lr_scheduler=args.lr_scheduler,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        warmup_prop=args.warmup_prop
    )

    decay = set()
    no_decay = set()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            decay.add(m.weight)
            if m.bias is not None:
                no_decay.add(m.bias)
        elif isinstance(m, nn.Embedding):
            decay.add(m.weight)
        elif isinstance(m, nn.LayerNorm):
            no_decay.add(m.weight)
            if m.bias is not None:
                no_decay.add(m.bias)

    assert len(list(model.parameters())) == len(decay) + len(no_decay)
    groups = [
        {'params': list(decay), 'weight_decay': optimizer.weight_decay},
        {'params': list(no_decay), 'weight_decay': 0.0}
    ]
    optimizer.set_parameters(groups)

    if checkpoint:
        optimizer.optimizer.load_state_dict(checkpoint['optim'])
        optimizer._step = checkpoint['step']
        if args.use_cuda:
            for state in optimizer.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if len(optimizer.optimizer.state) < 1:
            raise RuntimeError("Error: loaded Adam optimizer from existing model "
                               "but optimizer state is empty")

    return optimizer


class Optimizer(object):

    def __init__(self, method='adam', learning_rate=3, max_grad_norm=2.0, weight_decay=0.01,
                 eps=1e-9, beta1=0.9, beta2=0.999, model_size=None, lr_scheduler=None,
                 train_steps=None, warmup_steps=8000, warmup_prop=None):
        self.method = method
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.eps = eps
        self._step = 0
        self.betas = (beta1, beta2)
        self.model_size = model_size
        self.lr_scheduler = lr_scheduler
        self.train_steps = train_steps

        self.warmup_steps = warmup_steps
        if self.lr_scheduler == 'linear_warmup_decay':
            self.warmup_steps = self.train_steps * warmup_prop

        self.last_ppl = None
        self.start_decay = False

        self.params = []

        self.optimizer = None

    def set_parameters(self, groups):
        self.params = [[p for p in group['params'] if p.requires_grad] for group in groups]
        if self.method.lower() == 'adam':
            self.optimizer = optim.Adam(groups, lr=self.learning_rate, betas=self.betas, eps=self.eps)
        elif self.method.lower() == 'adamw':
            self.optimizer = optim.AdamW(groups, lr=self.learning_rate, betas=self.betas, eps=self.eps)
        else:
            raise NotImplementedError('Optimizer not implemented, please choose adam or adamw')

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def step(self):
        self._step += 1
        if self.lr_scheduler == "noam":
            self._set_rate(
                self.original_lr *
                (self.model_size ** -0.5 *
                 min(self._step ** -0.5, self._step * self.warmup_steps ** -1.5)))
        elif self.lr_scheduler == 'linear_warmup_decay':
            if self._step < self.warmup_steps:
                self._set_rate(self.original_lr * (self._step / self.warmup_steps))
            else:
                self._set_rate(
                    self.original_lr *
                    (1 - self._step / self.train_steps))

        if self.max_grad_norm > 0.0:
            for i in range(len(self.params)):
                clip_grad_norm_(self.params[i], self.max_grad_norm)
        self.optimizer.step()
