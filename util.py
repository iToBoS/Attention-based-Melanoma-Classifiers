# Fix Warmup Bug
from warmup_scheduler import GradualWarmupScheduler  # https://github.com/ildoonet/pytorch-gradual-warmup-lr
import numpy as np

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

class EarlyStopper:
    def __init__(self, patience=5, auc_delta=0.001, f1_delta=0.001):
        self.patience = patience
        self.auc_delta = auc_delta
        self.f1_delta = f1_delta
        self.counter = 0
        self.max_validation_auc = -np.inf
        self.max_validation_f1 = -np.inf

    def early_stop(self, validation_auc, validation_f1):
        if validation_auc > self.max_validation_auc:
            self.max_validation_auc = validation_auc
            self.counter = 0

        elif validation_f1 > self.max_validation_f1:
            self.max_validation_f1 = validation_f1
            self.counter = 0

        elif validation_auc < (self.max_validation_auc - self.auc_delta) or validation_f1 < (self.max_validation_f1 - self.f1_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
            
        return False