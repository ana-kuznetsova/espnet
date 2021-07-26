from typing import Any
from typing import Sequence
from typing import Union
from typing import List
from typing import Iterator
from copy import deepcopy

import numpy as np
from torch.utils.data import DataLoader
from typeguard import check_argument_types
import random

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.curriculum.curriculum_sampler import CurriculumSampler

from torch.utils.data.dataloader import default_collate


class TaskDataLoader:

    def __init__(
        self,
        dataset,
        batch_sampler,
        collate_fn = default_collate,
    ):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.batch_iter = iter(batch_sampler)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        batch = next(self.batch_iter)
        return self.collate_fn([self.dataset[i] for i in batch])

    next = __next__


class CurriculumIterFactory(AbsIterFactory):
    def __init__(
        self,
        dataset,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        num_iters_per_epoch: int = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):

        assert check_argument_types()
        
        self.sampler = batches
        self.K = len(batches)
        self.dataset = dataset
        self.num_iters_per_epoch = num_iters_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory
        self.loaders = None

    
    def build_iter(self, epoch=1):
        #epoch is a dummy variable to accomodate trainer.run() method.
        #Instead of one data loader we return K data loader for each task
        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        self.loaders = []
        for i in range(len(self.sampler)):
            random.shuffle(self.sampler[i])
            self.loaders.append(
               TaskDataLoader(
                   dataset=self.dataset,
                   batch_sampler=self.sampler[i],
                   **kwargs,
               )
           )
        return self.loaders

    def refill_task(self, k):
        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        random.shuffle(self.sampler[k])
        self.loaders[k] = TaskDataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler[k],
            **kwargs,
        )
        
        return self.loaders[k]
