import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def build_optim(args, model, checkpoint):
    optimizer = Optimizer(
        args.optimizer, args.lr, args.max_grad_norm,
        weight_decay=args.weight_decay,
        eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
        lr_scheduler=args.lr_scheduler,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_prop * args.train_steps,
        model_size=args.hidden_size
    )

    optimizer.set_parameters(list(model.named_parameters()))

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

    def __init__(self, method='adam', learning_rate=3, max_grad_norm=2.0, weight_decay=0.01, eps=1e-9,
                 beta1=0.9, beta2=0.999, lr_scheduler=None, train_steps=None, warmup_steps=None, model_size=None):
        self.method = method
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.eps = eps
        self._step = 0
        self.betas = (beta1, beta2)
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.train_steps = train_steps

        self.last_ppl = None
        self.start_decay = False

        self.params = []
        self.sparse_params = []

        self.optimizer = None

    def set_parameters(self, params):
        for k, p in params:
            if p.requires_grad:
                self.params.append(p)
        self.optimizer = optim.Adam(self.params, lr=self.learning_rate, betas=self.betas,
                                    weight_decay=self.weight_decay, eps=self.eps)

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer.param_groups[0]['lr'] = self.learning_rate

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

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
