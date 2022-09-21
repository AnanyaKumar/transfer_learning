
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np

# Code adapter from Hugging Face transformers implementation.

class CosineScheduleWithWarmup(LambdaLR):
    """Linear warm up and then cosine annealing of the learning rate."""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        super(CosineScheduleWithWarmup, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))


class LayerWiseSchedule(LambdaLR):
    """Linear warm up and then cosine annealing of the learning rate."""
    def __init__(self, optimizer, num_epochs, warmup_epochs=0, cooldown_epochs=0, decay_exp=3.73, cosine=False, num_cycles=0.5, last_epoch=-1):
        self.num_epochs = num_epochs
        self.num_cycles = num_cycles
        self.cosine = cosine
        self.decay_exp = decay_exp
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        super(LayerWiseSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_epoch):
        effective_curr_epoch = max(0, current_epoch - self.warmup_epochs)
        effective_total_epochs = max(1, self.num_epochs - self.warmup_epochs - self.cooldown_epochs)
        if current_epoch >= self.num_epochs - self.cooldown_epochs or effective_total_epochs == 1:
            return 1.0
        progress = float(effective_curr_epoch) / float(effective_total_epochs - 1.0)
        # num_layers_tuning = int(progress * (num_layers + 1))
        # assert num_layers_tuning >= 0
        # if num_layers_tuning > num_layers:
        #     num_layers_tuning = num_layers
        lr_multiplier = np.exp(self.decay_exp * (1.0 - progress))
        if self.cosine:
            lr_multiplier *= max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))
        return lr_multiplier



