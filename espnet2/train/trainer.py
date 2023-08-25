"""Trainer module."""
import argparse
<<<<<<< HEAD
import sys
from contextlib import contextmanager
=======
>>>>>>> bcd20948db7846ee523443ef9fd78c7a1248c95e
import dataclasses
import logging
import time
<<<<<<< HEAD
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Any
=======
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
>>>>>>> bcd20948db7846ee523443ef9fd78c7a1248c95e

import humanfriendly
import numpy as np
import torch
import torch.nn
import torch.optim
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.utils.build_dataclass import build_dataclass
<<<<<<< HEAD
from espnet2.curriculum.curriculum_generator import AbsCurriculumGenerator
from espnet2.curriculum.curriculum_generator import EXP3SCurriculumGenerator, SWUCBCurriculumGenerator, ManualCurriculumGenerator
from espnet2.curriculum.curriculum_iter_factory import CurriculumIterFactory
=======
from espnet2.utils.kwargs2args import kwargs2args
>>>>>>> bcd20948db7846ee523443ef9fd78c7a1248c95e

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

autocast_args = dict()
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast

    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        autocast_args = dict(dtype=torch.bfloat16)
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None

import torch
import sys

@dataclasses.dataclass
class TrainerOptions:
    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_matplotlib: bool
    use_tensorboard: bool
    use_wandb: bool
    output_dir: Union[Path, str]
    max_epoch: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    use_curriculum: bool
    curriculum_algo: Optional[str]
    gain_type: Optional[str]
    refill_task: Optional[bool]
    gen_log_dir: Optional[str]
    hist_size: Optional[int]
    threshold: Optional[float]
    gamma: Optional[float]
    lmbda_slow: Optional[float]
    lmbda_fast: Optional[float]
    slow_k: Optional[float]
    epsilon: Optional[float]
    eta: Optional[float]
    beta: Optional[float]
    start_curriculum: Optional[int]
    wandb_model_log_interval: int
<<<<<<< HEAD
    man_curr_file: Optional[str]
    epochs_per_stage: Optional[int]
=======
    create_graph_in_tensorboard: bool
>>>>>>> bcd20948db7846ee523443ef9fd78c7a1248c95e


class Trainer:
    """Trainer having a optimizer.

    If you'd like to use multiple optimizers, then inherit this class
    and override the methods if necessary - at least "train_one_epoch()"

    >>> class TwoOptimizerTrainer(Trainer):
    ...     @classmethod
    ...     def add_arguments(cls, parser):
    ...         ...
    ...
    ...     @classmethod
    ...     def train_one_epoch(cls, model, optimizers, ...):
    ...         loss1 = model.model1(...)
    ...         loss1.backward()
    ...         optimizers[0].step()
    ...
    ...         loss2 = model.model2(...)
    ...         loss2.backward()
    ...         optimizers[1].step()

    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        assert check_argument_types()
        return build_dataclass(TrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Reserved for future development of another Trainer"""
        pass

    @staticmethod
    def resume(
        checkpoint: Union[str, Path],
        model: torch.nn.Module,
        reporter: Reporter,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        ngpu: int = 0,
    ):
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
        )
        model.load_state_dict(states["model"])
        reporter.load_state_dict(states["reporter"])
        for optimizer, state in zip(optimizers, states["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, states["schedulers"]):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is None:
                logging.warning("scaler state is not found")
            else:
                scaler.load_state_dict(states["scaler"])

        logging.info(f"The training was resumed using {checkpoint}")

    @classmethod
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        trainer_options,
        distributed_option: DistributedOption,
    ) -> None:
        """Perform training. This method performs the main process of training."""
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        assert is_dataclass(trainer_options), type(trainer_options)
        assert len(optimizers) == len(schedulers), (len(optimizers), len(schedulers))

        if isinstance(trainer_options.keep_nbest_models, int):
            keep_nbest_models = [trainer_options.keep_nbest_models]
        else:
            if len(trainer_options.keep_nbest_models) == 0:
                logging.warning("No keep_nbest_models is given. Change to [1]")
                trainer_options.keep_nbest_models = [1]
            keep_nbest_models = trainer_options.keep_nbest_models

        output_dir = Path(trainer_options.output_dir)

        reporter = Reporter()
        if trainer_options.use_amp:
            if V(torch.__version__) < V("1.6.0"):
                raise RuntimeError(
                    "Require torch>=1.6.0 for  Automatic Mixed Precision"
                )
            if trainer_options.sharded_ddp:
                if fairscale is None:
                    raise RuntimeError(
                        "Requiring fairscale. Do 'pip install fairscale'"
                    )
                scaler = fairscale.optim.grad_scaler.ShardedGradScaler()
            else:
                scaler = GradScaler()
        else:
            scaler = None

        #Handle curriculum pretraining
        if (trainer_options.start_curriculum > 0) and (output_dir / f"checkpoint_{trainer_options.start_curriculum}.pth").exists():
            cls.resume(
                checkpoint=output_dir / f"checkpoint_{trainer_options.start_curriculum}.pth",
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                reporter=reporter,
                scaler=scaler,
                ngpu=trainer_options.ngpu,
            )


        if trainer_options.resume and (output_dir / "checkpoint.pth").exists() and (trainer_options.start_curriculum==0):
            cls.resume(
                checkpoint=output_dir / "checkpoint.pth",
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                reporter=reporter,
                scaler=scaler,
                ngpu=trainer_options.ngpu,
            )
        
        start_epoch = reporter.get_epoch() + 1

        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        if distributed_option.distributed:
            if trainer_options.sharded_ddp:
                dp_model = fairscale.nn.data_parallel.ShardedDataParallel(
                    module=model,
                    sharded_optimizer=optimizers,
                )
            else:
                dp_model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=(
                        # Perform multi-Process with multi-GPUs
                        [torch.cuda.current_device()]
                        if distributed_option.ngpu == 1
                        # Perform single-Process with multi-GPUs
                        else None
                    ),
                    output_device=(
                        torch.cuda.current_device()
                        if distributed_option.ngpu == 1
                        else None
                    ),
                    find_unused_parameters=trainer_options.unused_parameters,
                )
        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if trainer_options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            from torch.utils.tensorboard import SummaryWriter

            train_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "train")
            )
            valid_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "valid")
            )
        else:
            train_summary_writer = None

        start_time = time.perf_counter()

        #### Initialise Curriculum Learning Environment #######
        if trainer_options.use_curriculum==True:
            #wandb.init(project='curriculum_learning_2.0', entity='anakuzne')
            #wandb.watch(model)

            restore_curriculum = False
            if start_epoch > 1:
                restore_curriculum = True

            #restore_curriculum = True
        
            if trainer_options.curriculum_algo=='exp3s':
                curriculum_generator = EXP3SCurriculumGenerator(
                                            K=train_iter_factory.K,
                                            init='zeros',
                                            hist_size=trainer_options.hist_size,
                                            log_dir=str(output_dir),
                                            gain_type=trainer_options.gain_type,
                                            restore=restore_curriculum,
                                            iepoch=start_epoch,
                                            epsilon=trainer_options.epsilon,
                                            eta=trainer_options.eta,
                                            beta=trainer_options.beta,
                                            )
            elif trainer_options.curriculum_algo=='swucb':
                curriculum_generator = SWUCBCurriculumGenerator(
                                       K=train_iter_factory.K,
                                       hist_size=trainer_options.hist_size,
                                       log_dir=str(output_dir),
                                       lmbda_slow=trainer_options.lmbda_slow,
                                       lmbda_fast=trainer_options.lmbda_fast,
                                       threshold=trainer_options.threshold,
                                       gamma=trainer_options.gamma,
                                       slow_k=trainer_options.slow_k,
                                       restore=restore_curriculum,
                                       gain_type=trainer_options.gain_type,
                                       iepoch=start_epoch,
                )
            elif trainer_options.curriculum_algo=='manual':
                curriculum_generator = ManualCurriculumGenerator(K=train_iter_factory.K,
                                                                 man_curr_file=trainer_options.man_curr_file,
                                                                 epochs_per_stage=trainer_options.epochs_per_stage,
                                                                 log_dir=str(output_dir),
                                                                 restore=restore_curriculum,
                                                                 iepoch=start_epoch,
                                                                 )

        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        trainer_options.max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (trainer_options.max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{trainer_options.max_epoch} epoch started")
            set_all_random_seed(trainer_options.seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
<<<<<<< HEAD
                if trainer_options.use_curriculum==True:

                    if (iepoch==1) or (trainer_options.resume)==True:
                        trainer_options.resume=False
                        logging.info(f"Loading data for iterators...") 
                        tasks = train_iter_factory.build_iter(iepoch)

                    if trainer_options.gain_type=='VPG':

                        all_steps_are_invalid, train_iter_factory, valid_iter_factory, tasks = cls.train_one_epoch_curriculum(
                            model=dp_model,
                            optimizers=optimizers,
                            schedulers=schedulers,
                            iterator=train_iter_factory,
                            tasks=tasks,
                            reporter=sub_reporter,
                            curriculum_generator=curriculum_generator,   
                            scaler=scaler,
                            summary_writer=summary_writer,
                            options=trainer_options,
                            distributed_option=distributed_option,
                            iepoch=iepoch,
                            valid_iterator=valid_iter_factory,
                        )
                    else:    

                        all_steps_are_invalid, train_iter_factory, tasks = cls.train_one_epoch_curriculum(
                                model=dp_model,
                                optimizers=optimizers,
                                schedulers=schedulers,
                                iterator=train_iter_factory,
                                tasks=tasks,
                                reporter=sub_reporter,
                                curriculum_generator=curriculum_generator,   
                                scaler=scaler,
                                summary_writer=summary_writer,
                                options=trainer_options,
                                distributed_option=distributed_option,
                                iepoch=iepoch,
                            )

                else:
                    all_steps_are_invalid = cls.train_one_epoch(
                        model=dp_model,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        iterator=train_iter_factory.build_iter(iepoch),
                        reporter=sub_reporter,
                        scaler=scaler,
                        summary_writer=summary_writer,
                        options=trainer_options,
                        distributed_option=distributed_option,
                    )
=======
                all_steps_are_invalid = cls.train_one_epoch(
                    model=dp_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=train_summary_writer,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )
>>>>>>> bcd20948db7846ee523443ef9fd78c7a1248c95e

            
            with reporter.observe("valid") as sub_reporter:
                cls.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )
            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # att_plot doesn't support distributed
                if plot_attention_iter_factory is not None:
                    with reporter.observe("att_plot") as sub_reporter:
                        cls.plot_attention(
                            model=model,
                            output_dir=output_dir / "att_ws",
                            summary_writer=train_summary_writer,
                            iterator=plot_attention_iter_factory.build_iter(iepoch),
                            reporter=sub_reporter,
                            options=trainer_options,
                        )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(
                        reporter.get_value(*trainer_options.val_scheduler_criterion)
                    )
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()
            if trainer_options.sharded_ddp:
                for optimizer in optimizers:
                    if isinstance(optimizer, fairscale.optim.oss.OSS):
                        optimizer.consolidate_state_dict()

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # 3. Report the results
                logging.info(reporter.log_message())
                if trainer_options.use_matplotlib:
                    reporter.matplotlib_plot(output_dir / "images")
                if train_summary_writer is not None:
                    reporter.tensorboard_add_scalar(train_summary_writer, key1="train")
                    reporter.tensorboard_add_scalar(valid_summary_writer, key1="valid")
                if trainer_options.use_wandb:
                    reporter.wandb_log()

                # 4. Save/Update the checkpoint
                torch.save(
                    {
                        "model": model.state_dict(),
                        "reporter": reporter.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [
                            s.state_dict() if s is not None else None
                            for s in schedulers
                        ],
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    },
                    output_dir / "checkpoint.pth",
                )

                if iepoch%5==0:
                    torch.save(
                    {
                        "model": model.state_dict(),
                        "reporter": reporter.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [
                            s.state_dict() if s is not None else None
                            for s in schedulers
                        ],
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    },
                    output_dir / f"checkpoint_{iepoch}.pth",
                )

                # 5. Save and log the model and update the link to the best model
                torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth")

                # Creates a sym link latest.pth -> {iepoch}epoch.pth
                p = output_dir / "latest.pth"
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(f"{iepoch}epoch.pth")

                _improved = []
                for _phase, k, _mode in trainer_options.best_model_criterion:
                    # e.g. _phase, k, _mode = "train", "loss", "min"
                    if reporter.has(_phase, k):
                        best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                        # Creates sym links if it's the best result
                        if best_epoch == iepoch:
                            p = output_dir / f"{_phase}.{k}.best.pth"
                            if p.is_symlink() or p.exists():
                                p.unlink()
                            p.symlink_to(f"{iepoch}epoch.pth")
                            _improved.append(f"{_phase}.{k}")
                if len(_improved) == 0:
                    logging.info("There are no improvements in this epoch")
                else:
                    logging.info(
                        "The best model has been updated: " + ", ".join(_improved)
                    )

                log_model = (
                    trainer_options.wandb_model_log_interval > 0
                    and iepoch % trainer_options.wandb_model_log_interval == 0
                )
                if log_model and trainer_options.use_wandb:
                    import wandb

                    logging.info("Logging Model on this epoch :::::")
                    artifact = wandb.Artifact(
                        name=f"model_{wandb.run.id}",
                        type="model",
                        metadata={"improved": _improved},
                    )
                    artifact.add_file(str(output_dir / f"{iepoch}epoch.pth"))
                    aliases = [
                        f"epoch-{iepoch}",
                        "best" if best_epoch == iepoch else "",
                    ]
                    wandb.log_artifact(artifact, aliases=aliases)

                # 6. Remove the model files excluding n-best epoch and latest epoch
                _removed = []
                # Get the union set of the n-best among multiple criterion
                nbests = set().union(
                    *[
                        set(reporter.sort_epochs(ph, k, m)[: max(keep_nbest_models)])
                        for ph, k, m in trainer_options.best_model_criterion
                        if reporter.has(ph, k)
                    ]
                )

                # Generated n-best averaged model
                if (
                    trainer_options.nbest_averaging_interval > 0
                    and iepoch % trainer_options.nbest_averaging_interval == 0
                ):
                    average_nbest_models(
                        reporter=reporter,
                        output_dir=output_dir,
                        best_model_criterion=trainer_options.best_model_criterion,
                        nbest=keep_nbest_models,
                        suffix=f"till{iepoch}epoch",
                    )

                for e in range(1, iepoch):
                    p = output_dir / f"{e}epoch.pth"
                    if p.exists() and e not in nbests:
                        p.unlink()
                        _removed.append(str(p))
                if len(_removed) != 0:
                    logging.info("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    "The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if trainer_options.patience is not None:
                if reporter.check_early_stopping(
                    trainer_options.patience, *trainer_options.early_stopping_criterion
                ):
                    break

        else:
            logging.info(
                f"The training was finished at {trainer_options.max_epoch} epochs "
            )

        # Generated n-best averaged model
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=trainer_options.best_model_criterion,
                nbest=keep_nbest_models,
            )
    @classmethod
    def train_one_batch(cls,
                        batch,
                        model,
                        scaler,
                        ngpu,
                        distributed,
                        reporter,
                        iiter,
                        accum_grad,
                        grad_noise,
                        grad_clip,
                        grad_clip_type,
                        optimizers, 
                        schedulers,
                        start_time
                        ):
        model.train()
        with autocast(scaler is not None):
            retval = model(**batch)
            loss, stats, weight = retval
            optim_idx = None
            stats = {k: v for k, v in stats.items() if v is not None}
            if ngpu > 1 or distributed:
                # Apply weighted averaging for loss and stats
                loss = (loss * weight.type(loss.dtype)).sum()

                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

                # Now weight is summation over all workers
                loss /= weight
            if distributed:
                # NOTE(kamo): Multiply world_size because DistributedDataParallel
                # automatically normalizes the gradient by world_size.
                loss *= torch.distributed.get_world_size()

            loss /= accum_grad
        #reporter.next()
        all_steps_are_invalid = True
        #reporter.register(stats, weight)
        with reporter.measure_time("backward_time"):
            if scaler is not None:
                # Scales loss.  Calls backward() on scaled loss
                # to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose
                # for corresponding forward ops.
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss.detach()
            torch.cuda.empty_cache()
            
            if iiter % accum_grad == 0:
                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                # gradient noise injection
                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                # PyTorch<=1.4, clip_grad_norm_ returns float value
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )
                    
                    # Must invoke scaler.update() if unscale_() is used in the iteration
                    # to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                else:
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                # scaler.step() first unscales the gradients of
                                # the optimizer's assigned params.
                                scaler.step(optimizer)
                                # Updates the scale for next iteration.
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
                            optimizer.zero_grad()

                reporter.register(stats, weight)
                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    dict(
                        {
                            f"optim{i}_lr{j}": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
            
        return all_steps_are_invalid
                

    @classmethod
    def get_loss_eval_mode(cls,
                            batch,
                            model,
                            scaler,
                            ngpu,
                            distributed,
                            reporter,
                            iiter,
                            accum_grad):


        model.eval()
        with autocast(scaler is not None):
            with torch.no_grad():
                retval = model(**batch)

                loss, stats, weight = retval

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()
                    
                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)
                    
                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad
                loss = loss.detach()
        return loss
    """
    @classmethod
    def train_one_epoch_curriculum(
        cls,
        model: torch.nn.Module,
        iterator: CurriculumIterFactory,
        tasks: List,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        curriculum_generator: AbsCurriculumGenerator,
        summary_writer: Optional[SummaryWriter],
        options: TrainerOptions,
        distributed_option: DistributedOption,
        iepoch: int,
        **kwargs,
    ) -> bool:
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        distributed = distributed_option.distributed
        loss_before = 0
        loss_after = 0
        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        tasks = [iter(it) for it in tasks]

        if options.gain_type=='VPG':
            valid_iterator = kwargs["valid_iterator"]
            valid_task = iter(valid_iterator.build_iter(iepoch))

        iiter = 0
        #Reset the exausted tasks list
        curriculum_generator.reset_exhausted() 
        k = np.random.choice(curriculum_generator.K)
        logging.info(f"Start k:{k}")
        while iiter < iterator.num_iters_per_epoch:
            iiter+=1

            # For pretraining select task from a uniform distribution
            if (options.start_curriculum > 0) and (iepoch < options.start_curriculum):
                arr = np.arange(curriculum_generator.K)
                probs = np.ones(curriculum_generator.K)/len(arr)
                k = int(np.random.choice(arr, size=1, p=probs))
            else:
                if iiter % accum_grad == 0:
                    k = curriculum_generator.get_next_task_ind(iiter=iiter, iepoch=iepoch)

            try:
                _, batch = tasks[k].next()
            except StopIteration as e:
                if options.refill_task==True:
                    logging.info(f"Refilled task {k}.")
                    tasks.pop(k)
                    tasks.insert(k, iter(iterator.refill_task(k)))
                    _, batch = tasks[k].next()
                else:   
                    curriculum_generator.report_exhausted_task(k)
                    tasks.pop(k)
                    tasks.insert(k, iter(iterator.refill_task(k)))
                    logging.info(f"Task {k} is exhausted.")
                    if curriculum_generator.all_exhausted():                     
                        curriculum_generator.reset_exhausted()
                        break
                    iiter -= 1
                    continue
            
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            if no_forward_run:
                all_steps_are_invalid = False
                continue

            if options.gain_type=='PG':
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
                #Calculate loss before training on the batch
                loss1 = cls.get_loss_eval_mode(
                        batch,
                        model,
                        scaler,
                        ngpu,
                        distributed,
                        reporter,
                        iiter,
                        accum_grad 
                )


                all_steps_are_invalid = cls.train_one_batch(
                                            batch,
                                            model,
                                            scaler,
                                            ngpu,
                                            distributed,
                                            reporter,
                                            iiter,
                                            accum_grad,
                                            grad_noise,
                                            grad_clip,
                                            grad_clip_type,
                                            optimizers,
                                            schedulers,
                                            start_time
                                            )
                loss2 = cls.get_loss_eval_mode(
                        batch,
                        model,
                        scaler,
                        ngpu,
                        distributed,
                        reporter,
                        iiter,
                        accum_grad 
                        )
            
            elif options.gain_type=='VPG':
                try:
                    _, batch_valid = valid_task.next()
                except StopIteration as e:
                    valid_task = iter(valid_iterator.build_iter(iepoch))
                    _, batch_valid = valid_task.next()

                batch_valid_gpu = to_device(batch_valid, "cuda" if ngpu > 0 else "cpu")
                loss1 = cls.get_loss_eval_mode(
                            batch_valid_gpu,
                            model,
                            scaler,
                            ngpu,
                            distributed,
                            reporter,
                            iiter,
                            accum_grad 
                            )
                del batch_valid_gpu

                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
                all_steps_are_invalid = cls.train_one_batch(
                                            batch,
                                            model,
                                            scaler,
                                            ngpu,
                                            distributed,
                                            reporter,
                                            iiter,
                                            accum_grad,
                                            grad_noise,
                                            grad_clip,
                                            grad_clip_type,
                                            optimizers,
                                            schedulers,
                                            start_time
                                            )

                batch_valid = to_device(batch_valid, "cuda" if ngpu > 0 else "cpu")
                loss2 = cls.get_loss_eval_mode(
                            batch_valid,
                            model,
                            scaler,
                            ngpu,
                            distributed,
                            reporter,
                            iiter,
                            accum_grad 
                            )

            elif options.gain_type=='SPG':
                #Sample second batch for evaluation
                try:
                    _, batch_eval = tasks[k].next()
                except StopIteration as e:
                    if options.refill_task==True:
                        logging.info(f"Refilled task {k}.")
                        tasks.pop(k)
                        tasks.insert(k, iter(iterator.refill_task(k)))
                        _, batch_eval = tasks[k].next()
                    #Add else condition for exhaust task option

                batch_eval_gpu = to_device(batch_eval, "cuda" if ngpu > 0 else "cpu")
                
                loss1 = cls.get_loss_eval_mode(
                            batch_eval_gpu,
                            model,
                            scaler,
                            ngpu,
                            distributed,
                            reporter,
                            iiter,
                            accum_grad 
                            )
                del batch_eval_gpu
               
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
                all_steps_are_invalid = cls.train_one_batch(
                                            batch,
                                            model,
                                            scaler,
                                            ngpu,
                                            distributed,
                                            reporter,
                                            iiter,
                                            accum_grad,
                                            grad_noise,
                                            grad_clip,
                                            grad_clip_type,
                                            optimizers,
                                            schedulers,
                                            start_time
                                            )

                batch_eval = to_device(batch_eval, "cuda" if ngpu > 0 else "cpu")
                loss2 = cls.get_loss_eval_mode(
                            batch_eval,
                            model,
                            scaler,
                            ngpu,
                            distributed,
                            reporter,
                            iiter,
                            accum_grad 
                            ) 
            elif options.curriculum_algo=='manual':
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
                all_steps_are_invalid = cls.train_one_batch(
                                            batch,
                                            model,
                                            scaler,
                                            ngpu,
                                            distributed,
                                            reporter,
                                            iiter,
                                            accum_grad,
                                            grad_noise,
                                            grad_clip,
                                            grad_clip_type,
                                            optimizers,
                                            schedulers,
                                            start_time
                                            )
          
            if iiter % accum_grad == 0:
                if not (np.isinf(loss1.item()) or np.isinf(loss2.item())):
                    loss_before = loss1.item()
                    loss_after = loss2.item()
            
                #if options.curriculum_algo!='manual' and not (np.isinf(loss1.item()) or np.isinf(loss2.item())):
                if options.curriculum_algo!='manual':
                    curriculum_generator.update_policy(
                        iepoch=iepoch,
                        iiter=iiter,
                        k=k, 
                        losses=(loss_before, loss_after), 
                        #losses=(loss1, loss2)
                        batch_lens=batch['speech_lengths'].detach().cpu().numpy(),
                        algo=options.curriculum_algo,
                        start_curriculum=options.start_curriculum,
                        gain_type=options.gain_type,
                    )
                else:
                    curriculum_generator.update_policy(iepoch, iiter, algo='manual', k=k)
            

                start_time = time.perf_counter()

                # NOTE(kamo): Call log_message() after next()
                reporter.next()
                if iiter % log_interval == 0:
                    logging.info(reporter.log_message(-log_interval))
                    if summary_writer is not None:
                        reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                    if use_wandb:
                        reporter.wandb_log()

                torch.cuda.empty_cache()            

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        logging.info(f"Finished epoch {iepoch}")

        if options.gain_type=="VPG":
            return all_steps_are_invalid, iterator, valid_iterator, tasks
        return all_steps_are_invalid, iterator, tasks

"""

    @classmethod
    def train_one_epoch_curriculum(
        cls,
        model: torch.nn.Module,
        iterator: CurriculumIterFactory,
        tasks: List,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        curriculum_generator: AbsCurriculumGenerator,
        summary_writer: Optional[SummaryWriter],
        options: TrainerOptions,
        distributed_option: DistributedOption,
        iepoch,
        **kwargs
    ) -> bool:
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        distributed = distributed_option.distributed

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        start_time = time.perf_counter()
        tasks = [iter(it) for it in tasks]

        if options.gain_type=='VPG':
            valid_iterator = kwargs["valid_iterator"]
            valid_task = iter(valid_iterator.build_iter(iepoch))

        iiter = 0
        #Reset the exausted tasks list
        curriculum_generator.reset_exhausted() 
        
        #counter for making sure that first K updates use the K tasks respectively.
        updates = 0

        #for iiter, (_, batch) in enumerate(
        #    reporter.measure_iter_time(iterator, "iter_time"), 1
        #):
        while iiter < iterator.num_iters_per_epoch:
            # For pretraining select task from a uniform distribution
            if (options.start_curriculum > 0) and (iepoch < options.start_curriculum):
                arr = np.arange(curriculum_generator.K)
                probs = np.ones(curriculum_generator.K)/len(arr)
                k = int(np.random.choice(arr, size=1, p=probs))
            else:
                if iiter % accum_grad == 0:
                    if updates < curriculum_generator.K:
                        k = curriculum_generator.get_next_task_ind(iiter=updates, iepoch=iepoch)
                        updates +=1
                    else:
                        k = curriculum_generator.get_next_task_ind(iiter=iiter, iepoch=iepoch)
            
            iiter += 1
            try:
                _, batch = tasks[k].next()
            except StopIteration as e:
                if options.refill_task==True:
                    logging.info(f"Refilled task {k}.")
                    tasks.pop(k)
                    tasks.insert(k, iter(iterator.refill_task(k)))
                    _, batch = tasks[k].next()
                else:   
                    curriculum_generator.report_exhausted_task(k)
                    tasks.pop(k)
                    tasks.insert(k, iter(iterator.refill_task(k)))
                    logging.info(f"Task {k} is exhausted.")
                    if curriculum_generator.all_exhausted():                     
                        curriculum_generator.reset_exhausted()
                        break
                    iiter -= 1
                    continue

            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            #batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

            if no_forward_run:
                all_steps_are_invalid = False
                continue

            if options.gain_type=='SPG':
                #Sample second batch for evaluation
                try:
                    _, batch_eval = tasks[k].next()
                except StopIteration as e:
                    if options.refill_task==True:
                        logging.info(f"Refilled task {k}.")
                        tasks.pop(k)
                        tasks.insert(k, iter(iterator.refill_task(k)))
                        _, batch_eval = tasks[k].next()
                    #Add else condition for exhaust task option
                
                if iiter % accum_grad == 0:

                    batch_eval_gpu = to_device(batch_eval, "cuda" if ngpu > 0 else "cpu")
                    loss1 = cls.get_loss_eval_mode(
                                batch_eval_gpu,
                                model,
                                scaler,
                                ngpu,
                                distributed,
                                reporter,
                                iiter,
                                accum_grad 
                                )
                    del batch_eval_gpu
               
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                with autocast(scaler is not None):
                    with reporter.measure_time("forward_time"):
                        retval = model(**batch)

                        # Note(kamo):
                        # Supporting two patterns for the returned value from the model
                        #   a. dict type
                        if isinstance(retval, dict):
                            loss = retval["loss"]
                            stats = retval["stats"]
                            weight = retval["weight"]
                            optim_idx = retval.get("optim_idx")
                            if optim_idx is not None and not isinstance(optim_idx, int):
                                if not isinstance(optim_idx, torch.Tensor):
                                    raise RuntimeError(
                                        "optim_idx must be int or 1dim torch.Tensor, "
                                        f"but got {type(optim_idx)}"
                                    )
                                if optim_idx.dim() >= 2:
                                    raise RuntimeError(
                                        "optim_idx must be int or 1dim torch.Tensor, "
                                        f"but got {optim_idx.dim()}dim tensor"
                                    )
                                if optim_idx.dim() == 1:
                                    for v in optim_idx:
                                        if v != optim_idx[0]:
                                            raise RuntimeError(
                                                "optim_idx must be 1dim tensor "
                                                "having same values for all entries"
                                            )
                                    optim_idx = optim_idx[0].item()
                                else:
                                    optim_idx = optim_idx.item()

                        #   b. tuple or list type
                        else:
                            loss, stats, weight = retval
                            optim_idx = None

                    stats = {k: v for k, v in stats.items() if v is not None}
                    if ngpu > 1 or distributed:
                        # Apply weighted averaging for loss and stats
                        loss = (loss * weight.type(loss.dtype)).sum()

                        # if distributed, this method can also apply all_reduce()
                        stats, weight = recursive_average(stats, weight, distributed)

                        # Now weight is summation over all workers
                        loss /= weight
                    if distributed:
                        # NOTE(kamo): Multiply world_size because DistributedDataParallel
                        # automatically normalizes the gradient by world_size.
                        loss *= torch.distributed.get_world_size()

                    loss /= accum_grad

                reporter.register(stats, weight)
                
                with reporter.measure_time("backward_time"):
                    if scaler is not None:
                        # Scales loss.  Calls backward() on scaled loss
                        # to create scaled gradients.
                        # Backward passes under autocast are not recommended.
                        # Backward ops run in the same dtype autocast chose
                        # for corresponding forward ops.
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if iiter % accum_grad == 0:
                    if scaler is not None:
                        # Unscales the gradients of optimizer's assigned params in-place
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.unscale_(optimizer)

                    # gradient noise injection
                    if grad_noise:
                        add_gradient_noise(
                            model,
                            reporter.get_total_count(),
                            duration=100,
                            eta=1.0,
                            scale_factor=0.55,
                        )

                    # compute the gradient norm to check if it is normal or not
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                        norm_type=grad_clip_type,
                    )
                    # PyTorch<=1.4, clip_grad_norm_ returns float value
                    if not isinstance(grad_norm, torch.Tensor):
                        grad_norm = torch.tensor(grad_norm)

                    if not torch.isfinite(grad_norm):
                        logging.warning(
                            f"The grad norm is {grad_norm}. Skipping updating the model."
                        )

                        # Must invoke scaler.update() if unscale_() is used in the iteration
                        # to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # Note that if the gradient has inf/nan values,
                        # scaler.step skips optimizer.step().
                        if scaler is not None:
                            for iopt, optimizer in enumerate(optimizers):
                                if optim_idx is not None and iopt != optim_idx:
                                    continue
                                scaler.step(optimizer)
                                scaler.update()

                    else:
                        all_steps_are_invalid = False
                        with reporter.measure_time("optim_step_time"):
                            for iopt, (optimizer, scheduler) in enumerate(
                                zip(optimizers, schedulers)
                            ):
                                if optim_idx is not None and iopt != optim_idx:
                                    continue
                                if scaler is not None:
                                    # scaler.step() first unscales the gradients of
                                    # the optimizer's assigned params.
                                    scaler.step(optimizer)
                                    # Updates the scale for next iteration.
                                    scaler.update()
                                else:
                                    optimizer.step()
                                if isinstance(scheduler, AbsBatchStepScheduler):
                                    scheduler.step()
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        optimizer.zero_grad()

                    batch_eval_gpu = to_device(batch_eval, "cuda" if ngpu > 0 else "cpu")
                    loss2 = cls.get_loss_eval_mode(
                                batch_eval_gpu,
                                model,
                                scaler,
                                ngpu,
                                distributed,
                                reporter,
                                iiter,
                                accum_grad 
                                )
                    del batch_eval_gpu


                    if not (np.isinf(loss1.item()) or np.isinf(loss2.item())):
                        loss_before = loss1.item()
                        loss_after = loss2.item()
                
                        #if options.curriculum_algo!='manual' and not (np.isinf(loss1.item()) or np.isinf(loss2.item())):
                        if options.curriculum_algo!='manual':
                            curriculum_generator.update_policy(
                                iepoch=iepoch,
                                iiter=iiter,
                                k=k, 
                                losses=(loss_before, loss_after), 
                                batch_lens=batch['speech_lengths'].detach().cpu().numpy(),
                                algo=options.curriculum_algo,
                                start_curriculum=options.start_curriculum,
                                gain_type=options.gain_type,
                            )
                        else:
                            curriculum_generator.update_policy(iepoch, iiter, algo='manual', k=k)

                    # Register lr and train/load time[sec/step],
                    # where step refers to accum_grad * mini-batch
                    reporter.register(
                        dict(
                            {
                                f"optim{i}_lr{j}": pg["lr"]
                                for i, optimizer in enumerate(optimizers)
                                for j, pg in enumerate(optimizer.param_groups)
                                if "lr" in pg
                            },
                            train_time=time.perf_counter() - start_time,
                        ),
                    )
                    start_time = time.perf_counter()
                    #logging.info(f'IITER:{iiter}, interval:{log_interval}')
                    # NOTE(kamo): Call log_message() after next()
                    if iiter % log_interval == 0:
                        logging.info(reporter.log_message(-log_interval))
                        if summary_writer is not None:
                            reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                        if use_wandb:
                            reporter.wandb_log()
                    torch.cuda.empty_cache()
                reporter.next()
            
                
                            


        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        return all_steps_are_invalid, iterator, tasks














    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        create_graph_in_tensorboard = options.create_graph_in_tensorboard
        distributed = distributed_option.distributed

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

            if no_forward_run:
                all_steps_are_invalid = False
                continue

            if (
                create_graph_in_tensorboard
                and iiter == 1
                and summary_writer is not None
            ):
                if distributed:
                    _model = getattr(model, "module")
                else:
                    _model = model
                    if _model is not None:
                        try:
                            _args = kwargs2args(_model.forward, batch)
                        except (ValueError, TypeError):
                            logging.warning(
                                "inpect.signature() is failed for the model. "
                                "The graph can't be added for tensorboard."
                            )
                        else:
                            try:
                                summary_writer.add_graph(
                                    _model, _args, use_strict_trace=False
                                )
                            except Exception:
                                logging.warning(
                                    "summary_writer.add_graph() "
                                    "is failed for the model. "
                                    "The graph can't be added for tensorboard."
                                )
                            del _args
                    else:
                        logging.warning(
                            "model.module is not found (This should be a bug.)"
                        )
                del _model

            with autocast(
                scaler is not None,
                **autocast_args,
            ):
                with reporter.measure_time("forward_time"):
                    retval = model(**batch)

                    # Note(kamo):
                    # Supporting two patterns for the returned value from the model
                    #   a. dict type
                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if not isinstance(optim_idx, torch.Tensor):
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {type(optim_idx)}"
                                )
                            if optim_idx.dim() >= 2:
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {optim_idx.dim()}dim tensor"
                                )
                            if optim_idx.dim() == 1:
                                for v in optim_idx:
                                    if v != optim_idx[0]:
                                        raise RuntimeError(
                                            "optim_idx must be 1dim tensor "
                                            "having same values for all entries"
                                        )
                                optim_idx = optim_idx[0].item()
                            else:
                                optim_idx = optim_idx.item()

                    #   b. tuple or list type
                    else:
                        loss, stats, weight = retval
                        optim_idx = None

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()

                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)

                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad

            reporter.register(stats, weight)
            
            with reporter.measure_time("backward_time"):
                if scaler is not None:
                    # Scales loss.  Calls backward() on scaled loss
                    # to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose
                    # for corresponding forward ops.
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if iiter % accum_grad == 0:
                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                # gradient noise injection
                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                # PyTorch<=1.4, clip_grad_norm_ returns float value
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )

                    # Must invoke scaler.update() if unscale_() is used in the iteration
                    # to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                else:
                    reporter.register(
                        {
                            "grad_norm": grad_norm,
                            "clip": torch.where(
                                grad_norm > grad_clip,
                                grad_norm.new_tensor(100),
                                grad_norm.new_tensor(0),
                            ),
                            "loss_scale": scaler.get_scale() if scaler else 1.0,
                        }
                    )
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                # scaler.step() first unscales the gradients of
                                # the optimizer's assigned params.
                                scaler.step(optimizer)
                                # Updates the scale for next iteration.
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
                for iopt, optimizer in enumerate(optimizers):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    optimizer.zero_grad()

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    dict(
                        {
                            f"optim{i}_lr{j}": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
                start_time = time.perf_counter()

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()
            torch.cuda.empty_cache()            


        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for utt_id, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            retval = model(**batch)
            if isinstance(retval, dict):
                stats = retval["stats"]
                weight = retval["weight"]
            else:
                _, stats, weight = retval
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

    @classmethod
    @torch.no_grad()
    def plot_attention(
        cls,
        model: torch.nn.Module,
        output_dir: Optional[Path],
        summary_writer,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        assert check_argument_types()
        import matplotlib

        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        model.eval()
        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )

            batch["utt_id"] = ids

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            # 1. Forwarding model and gathering all attentions
            #    calculate_all_attentions() uses single gpu only.
            att_dict = calculate_all_attentions(model, batch)

            # 2. Plot attentions: This part is slow due to matplotlib
            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):
                    if isinstance(att_w, torch.Tensor):
                        att_w = att_w.detach().cpu().numpy()

                    if att_w.ndim == 2:
                        att_w = att_w[None]
                    elif att_w.ndim == 4:
                        # In multispkr_asr model case, the dimension could be 4.
                        att_w = np.concatenate(
                            [att_w[i] for i in range(att_w.shape[0])], axis=0
                        )
                    elif att_w.ndim > 4 or att_w.ndim == 1:
                        raise RuntimeError(f"Must be 2, 3 or 4 dimension: {att_w.ndim}")

                    w, h = plt.figaspect(1.0 / len(att_w))
                    fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                    axes = fig.subplots(1, len(att_w))
                    if len(att_w) == 1:
                        axes = [axes]

                    for ax, aw in zip(axes, att_w):
                        ax.imshow(aw.astype(np.float32), aspect="auto")
                        ax.set_title(f"{k}_{id_}")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    if output_dir is not None:
                        p = output_dir / id_ / f"{k}.{reporter.get_epoch()}ep.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(p)

                    if summary_writer is not None:
                        summary_writer.add_figure(
                            f"{k}_{id_}", fig, reporter.get_epoch()
                        )

                    if options.use_wandb:
                        import wandb

                        wandb.log({f"attention plot/{k}_{id_}": wandb.Image(fig)})
            reporter.next()
