import warnings
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class OneCycleScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25., final_div_factor=1e4):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Calculate the initial and final learning rates
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / (div_factor * final_div_factor)
        
        # Calculate the step sizes for each phase
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        super().__init__(optimizer, -1)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
        
        if self.last_epoch == -1:
            return [self.initial_lr for _ in self.base_lrs]
        
        # Calculate current step
        curr_step = self.last_epoch
        
        if curr_step <= self.step_size_up:
            # We're in the increasing phase
            pct = curr_step / self.step_size_up
            return [self.initial_lr + (self.max_lr - self.initial_lr) * pct for _ in self.base_lrs]
        else:
            # We're in the decreasing phase
            pct = (curr_step - self.step_size_up) / self.step_size_down
            return [self.max_lr - (self.max_lr - self.final_lr) * pct for _ in self.base_lrs]
            
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
