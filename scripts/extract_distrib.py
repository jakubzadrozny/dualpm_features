# RUN:
# python scripts/extract_distrib.py configs=configs/extract_feats.yml

from __future__ import annotations
import os

# Must be set before importing torch or any C-extensions that use OpenMP
os.environ["OMP_NUM_THREADS"] = "8"

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from dualpm_features.utils import parse_config

def worker(rank, world_size, config):
    device = f"cuda:{rank}"
    torch.set_num_threads(8)
    torch.cuda.set_device(device)
    
    run_config = OmegaConf.create({"_target_": "dualpm_features.extract.run", **config})
    hydra.utils.instantiate(
        run_config, 
        extractor={"device": device}, 
        rank=rank, 
        world_size=world_size
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    config = parse_config()
    
    n_gpus = torch.cuda.device_count()
    print(f"Running on {n_gpus} GPUs")
    
    if n_gpus > 1:
        mp.spawn(worker, args=(n_gpus, config), nprocs=n_gpus, join=True)
    else:
        worker(0, 1, config)
