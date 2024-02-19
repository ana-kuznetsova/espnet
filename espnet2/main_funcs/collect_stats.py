import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.forward_adaptor import ForwardAdaptor
from espnet2.train.abs_espnet_model import AbsESPnetModel

def adjust_codec_batch(batch:Dict[str, torch.Tensor], ngpu:int) -> Dict[str, torch.Tensor]:
    # Check divisibility
    speech = batch["speech"]
    speech_lengths = batch["speech_lengths"]

    codec_factor = 320

    logging.info("DEBUG adjust_bsize %s", speech.shape)

    if speech.shape[0] % ngpu != 0:
        speech = speech[:-1,]
        speech_lengths = speech_lengths[:-1]
    
    if (speech.shape[1] // codec_factor) % ngpu != 0:
        pass
    
    return batch


@torch.no_grad()
def collect_stats(
    model: Union[AbsESPnetModel, None],
    train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    valid_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    output_dir: Path,
    ngpu: Optional[int],
    log_interval: Optional[int],
    write_collected_feats: bool,
) -> None:
    """Perform on collect_stats mode.

    Running for deriving the shape information from data
    and gathering statistics.
    This method is used before executing train().

    """
    assert check_argument_types()

    npy_scp_writers = {}
    for itr, mode in zip([train_iter, valid_iter], ["train", "valid"]):
        if log_interval is None:
            try:
                log_interval = max(len(itr) // 20, 10)
            except TypeError:
                log_interval = 100

        sum_dict = defaultdict(lambda: 0)
        sq_dict = defaultdict(lambda: 0)
        count_dict = defaultdict(lambda: 0)

        with DatadirWriter(output_dir / mode) as datadir_writer:
            for iiter, (keys, batch) in enumerate(itr, 1):
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                # 1. Write shape file
                for name in batch:
                    if name.endswith("_lengths"):
                        continue
                    for i, (key, data) in enumerate(zip(keys, batch[name])):
                        if f"{name}_lengths" in batch:
                            lg = int(batch[f"{name}_lengths"][i])
                            data = data[:lg]
                        datadir_writer[f"{name}_shape"][key] = ",".join(
                            map(str, data.shape)
                        )

                if model is not None:
                    # 2. Extract feats
                    if ngpu <= 1:
                        data = model.collect_feats(**batch)
                    else:
                        # Note that data_parallel can parallelize only "forward()"
                        # Need a work around the data parallel bug in pytorch.
                        # Batch length should be divisible by ngpus
                        #logging.info("DEBUG batch brefore data_parallel %s", batch["speech"])
                        #batch = adjust_bsize(batch, ngpu)
                        data = data_parallel(
                            ForwardAdaptor(model, "collect_feats"),
                            (),
                            range(ngpu),
                            module_kwargs=batch,
                        )

                    # 3. Calculate sum and square sum
                    for key, v in data.items():
                        for i, (uttid, seq) in enumerate(zip(keys, v.cpu().numpy())):
                            # Truncate zero-padding region
                            if f"{key}_lengths" in data:
                                length = int(data[f"{key}_lengths"][i])
                                #feat_shapes =data[f"{key}"].shape
                                #print(f"DEBUG {feat_shapes}")
                                # seq: (Length, Dim, ...)
                                #print("DIM SEQ", seq.shape)
                                seq = seq[:length]
                                #print(seq.shape)

                            else:
                                # seq: (Dim, ...) -> (1, Dim, ...)
                                seq = seq[None]
                            # Accumulate value, its square, and count
                            #print(seq)
                            sum_dict[key] += seq.sum(0)
                            sq_dict[key] += (seq**2).sum(0)
                            count_dict[key] += len(seq)

                            # 4. [Option] Write derived features as npy format file.
                            if write_collected_feats:
                                # Instantiate NpyScpWriter for the first iteration
                                if (key, mode) not in npy_scp_writers:
                                    p = output_dir / mode / "collect_feats"
                                    npy_scp_writers[(key, mode)] = NpyScpWriter(
                                        p / f"data_{key}", p / f"{key}.scp"
                                    )
                                # Save array as npy file
                                npy_scp_writers[(key, mode)][uttid] = seq

                if iiter % log_interval == 0:
                    logging.info(f"Niter: {iiter}")

        for key in sum_dict:
            np.savez(
                output_dir / mode / f"{key}_stats.npz",
                count=count_dict[key],
                sum=sum_dict[key],
                sum_square=sq_dict[key],
            )

        # batch_keys and stats_keys are used by aggregate_stats_dirs.py
        with (output_dir / mode / "batch_keys").open("w", encoding="utf-8") as f:
            f.write(
                "\n".join(filter(lambda x: not x.endswith("_lengths"), batch)) + "\n"
            )
        with (output_dir / mode / "stats_keys").open("w", encoding="utf-8") as f:
            f.write("\n".join(sum_dict) + "\n")
