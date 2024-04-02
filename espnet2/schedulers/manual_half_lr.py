"""Manual half LR scheduler module."""
from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import check_argument_types
import logging
from torch.cuda.amp import GradScaler

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler

class ManuaHalflLR(_LRScheduler, AbsBatchStepScheduler):
    """HalfManualLR scheduler.
    Halves the learning rate if there is no improvement in validation
    loss for m number of epochs.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        no_improvement_target: int = -1,
        last_epoch: int = -1,
    ):
        assert check_argument_types()
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        self.no_improvement_target = no_improvement_target
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"no_improvement_target={self.no_improvement_target})"
        )

    def get_lr(self, no_improvement_current: int = -1) -> float:
        if no_improvement_current < self.no_improvement_target:
            lr = self.optimizer.param_groups[0]['lr']
        else:
            lr = self.optimizer.param_groups[0]['lr']/2
        logging.info("Current LR %s No improvement %s", lr, no_improvement_current)
        return [lr]
    
    def step(self, no_improvement_current:int = -1) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(no_improvement_current)):
            param_group["lr"] = lr