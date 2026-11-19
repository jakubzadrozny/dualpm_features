import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Must be set before importing torch or any C-extensions that use OpenMP
os.environ["OMP_NUM_THREADS"] = "8"

import faiss
import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from dualpm_features.extract import create_dataset_from_multiple, extract_features
from dualpm_features.utils import parse_config
from dualpm_features.wrappers.base import ExtractorWrapper

# create logger
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%I:%M:%S"
)
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


@dataclass
class Config:
    train_root: list[str]
    exp_name: str
    results_info_root: str
    pca_filename: str = "pca.faiss"
    img_postfix_train: list[str] = field(default_factory=lambda: ["_rgb.png"])
    mask_postfix_train: list[str] = field(default_factory=lambda: ["_mask.png"])
    batch_size: int = 4
    pca_dim: int = 64
    load_mask: bool = True
    normalize_features: bool = True
    debug_subset: bool = False
    shuffle_data: bool = True
    n_images_vis: int = 100
    n_train_random_samples: int = 4000


def train_and_save_pca(features: np.ndarray, pca_dim: int, pca_out_path: Path):
    logger.info(f"PCA on features with shape {features.shape}")
    features_dim = features.shape[1]
    pca_mat = faiss.PCAMatrix(features_dim, pca_dim)
    pca_mat.train(features)
    assert pca_mat.is_trained
    logger.info("PCA on features done")

    # save pca mat
    pca_out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_VectorTransform(pca_mat, str(pca_out_path))
    logger.info(f"PCA matrix saved to {pca_out_path}")


def run(
    extractor: ExtractorWrapper,
    args: Config,
    rank: int = 0,
    world_size: int = 1,
    tmp_dir: Optional[str] = None,
):
    exp_name = args.exp_name
    batch_size = args.batch_size
    pca_out_path = Path(args.results_info_root) / exp_name / args.pca_filename
    interpolation = 3

    feature_map_size = 60

    mask_transform = transforms.Compose(
        [
            transforms.Resize((feature_map_size, feature_map_size), interpolation),
            transforms.ToTensor(),
        ]
    )

    mask_postfix_train = args.mask_postfix_train if args.load_mask else None
    dataset = create_dataset_from_multiple(
        args.train_root,
        args.img_postfix_train,
        mask_postfix_train,
        n_random_samples=args.n_train_random_samples,
        debug_subset=args.debug_subset,
        transform=extractor.transform,
        mask_transform=mask_transform,
    )

    ind = list(range(len(dataset)))

    if world_size > 1:
        chunk_size = math.ceil(len(ind) / world_size)
        start = rank * chunk_size
        end = min(start + chunk_size, len(ind))
        local_ind = ind[start:end]

        if len(local_ind) > 0:
            local_dataset = torch.utils.data.Subset(dataset, local_ind)
            train_dataloader = DataLoader(
                local_dataset,
                batch_size=batch_size,
                shuffle=args.shuffle_data,
                num_workers=2,
                persistent_workers=True,
            )
        else:
            train_dataloader = None
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle_data,
            num_workers=2,
            persistent_workers=True,
        )

    if train_dataloader is not None:
        logger.info(f"Rank {rank}: Extracting features for PCA")
        features, masks = extract_features(
            args,
            train_dataloader,
            extractor,
            log_features=True,
            is_main_process=(rank == 0),
        )
        logger.info(f"Rank {rank}: Extracting features for PCA done")

        if args.load_mask:
            masks = masks > 0
            features = features[masks.squeeze(axis=-1)]

        if world_size > 1 and tmp_dir is not None:
            out_file = Path(tmp_dir) / f"features_{rank}.npy"
            np.save(out_file, features)
            return

    if world_size == 1:
        train_and_save_pca(features, args.pca_dim, pca_out_path)


def worker(rank, world_size, config_dict, tmp_dir):
    device = f"cuda:{rank}"
    torch.set_num_threads(8)
    torch.cuda.set_device(device)

    run_config = OmegaConf.create({"_target_": "dualpm_features.train_pca.run", **config_dict})
    hydra.utils.instantiate(
        run_config,
        extractor={"device": device},
        rank=rank,
        world_size=world_size,
        tmp_dir=tmp_dir,
    )


if __name__ == "__main__":
    config = parse_config()

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        mp.set_start_method("spawn", force=True)
        print(f"Running on {n_gpus} GPUs")

        config_dict = OmegaConf.to_container(config, resolve=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            mp.spawn(worker, args=(n_gpus, config_dict, tmp_dir), nprocs=n_gpus, join=True)

            logger.info("Loading extracted features from all GPUs")
            all_features = []
            for rank in range(n_gpus):
                feat_file = Path(tmp_dir) / f"features_{rank}.npy"
                if feat_file.exists():
                    all_features.append(np.load(feat_file))

            if len(all_features) > 0:
                features = np.concatenate(all_features, axis=0)
                exp_name = config.args.exp_name
                pca_out_path = Path(config.args.results_info_root) / exp_name / config.args.pca_filename

                train_and_save_pca(features, config.args.pca_dim, pca_out_path)
            else:
                logger.error("No features were extracted by any GPU!")
    else:
        run_config = OmegaConf.create({"_target_": "dualpm_features.train_pca.run", **config})
        hydra.utils.instantiate(run_config)
