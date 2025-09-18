import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import hydra
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms

from dualpm_features.extract import (
    create_dataloader,
    extract_features,
)
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
    name_depth: Optional[int] = 1
    img_postfix_train: list[str] = field(default_factory=lambda: ["_rgb.png"])
    mask_postfix_train: list[str] = field(default_factory=lambda: ["_mask.png"])
    batch_size: int = 12
    pca_dim: int = 64
    load_mask: bool = True
    normalize_features: bool = True
    debug_subset: bool = False
    shuffle_data: bool = True
    n_train_random_samples: int = 1000


def run(extractor: ExtractorWrapper, args: Config):
    exp_name = args.exp_name
    image_size = extractor.image_size
    batch_size = args.batch_size
    pca_out_path = Path(args.results_info_root) / exp_name / "pca.faiss"
    interpolation = 3

    mask_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation),
            transforms.ToTensor(),
        ]
    )

    mask_postfix_train = args.mask_postfix_train if args.load_mask else None
    train_dataloader = create_dataloader(
        args.train_root,
        batch_size,
        args.img_postfix_train,
        mask_postfix_train,
        n_random_samples=args.n_train_random_samples,
        debug_subset=args.debug_subset,
        shuffle=args.shuffle_data,
        transform=extractor.transform,
        mask_transform=mask_transform,
        name_depth=args.name_depth,
    )

    logger.info("Extracting features for PCA")
    features, masks = extract_features(
        args,
        train_dataloader,
        extractor,
        log_features=True,
    )
    logger.info("Extracting features for PCA done")

    if args.load_mask:
        masks = masks > 0
        features = features[rearrange(masks, "n c -> n")]

    logger.info("PCA on features")
    features_dim = features.shape[1]
    pca_dim = args.pca_dim
    pca_mat = faiss.PCAMatrix(features_dim, pca_dim)
    pca_mat.train(features)
    assert pca_mat.is_trained
    logger.info("PCA on features done")

    # save pca mat
    pca_out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_VectorTransform(pca_mat, str(pca_out_path))
    logger.info(f"PCA matrix saved to {pca_out_path}")


if __name__ == "__main__":
    config = parse_config()

    config = OmegaConf.create({"_target_": "__main__.run", **config})
    # Instantiate the function with arguments from the configuration
    hydra.utils.instantiate(config)
