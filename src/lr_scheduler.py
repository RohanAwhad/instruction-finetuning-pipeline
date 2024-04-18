import math
import torch.optim as optim

class CustomLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        step_size: int,
        n_warmup: int,
        n_total_iter: int,
        initial_lr: float,
        verbose: bool=False,
        last_epoch = -1
    ):
        self.step_size = step_size
        self.n_warmup = n_warmup
        self.n_total_iter = n_total_iter
        self.initial_lr = initial_lr
        self.verbose = verbose

        super().__init__(optimizer, last_epoch, verbose)
        self.step(1)  # to set lr_mult

    def step(self, global_step=0, epoch=None):
        total_tokens_proc = global_step * self.step_size
        if total_tokens_proc < self.n_warmup:
            self.lr_mult = float(total_tokens_proc) / float(max(1, self.n_warmup))
        else:
            progress = float(total_tokens_proc - self.n_warmup) / float(
                max(1, self.n_total_iter - self.n_warmup)
            )
            self.lr_mult = max( 
                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
            )
        if self.verbose:
            print('Setting LR_MULT to', self.lr_mult)

        super().step(epoch)

    def get_lr(self):
        lr_mult = self.lr_mult if self.lr_mult else 1
        return [self.initial_lr * lr_mult for _ in self.optimizer.param_groups]

