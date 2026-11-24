import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import hydra
import numpy as np
import PIL
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dualpm_features.feature_storage import save_feat_as_img
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


def arrange(images):
    rows = []
    for row in images:
        rows += [np.concatenate(row, axis=1)]
    image = np.concatenate(rows, axis=0)
    return image


def fast_scan(root, postfix):
    paths = []
    idx = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(postfix):            
                if idx % 1000 == 0:
                    print(idx, entry.path)
                idx += 1
                paths.append(Path(entry.path))
    return sorted(paths)


def create_dataset(
    root,
    debug_subset=False,
    n_random_samples=None,
    img_postfix="_rgb.jpg",
    mask_postfix=None,
    transform=None,
    mask_transform=None,
    samples_range=None,
):
    img_root = Path(root) / "renders"
    if not img_root.is_dir():
        img_root = Path(root) / "images"
    img_paths = fast_scan(str(img_root), img_postfix)
    dataset = FilesListDataset(
        root_dir=root,
        files_list=img_paths,
        img_postfix=img_postfix,
        mask_postfix=mask_postfix,
        transform=transform,
        mask_transform=mask_transform,
    )
    if samples_range is not None:
        indices = np.arange(samples_range[0], samples_range[1])
        dataset = torch.utils.data.Subset(dataset, indices)
    if n_random_samples:
        dataset = torch.utils.data.Subset(
            dataset, np.random.choice(len(dataset), n_random_samples)
        )
    if debug_subset:
        dataset = torch.utils.data.Subset(dataset, np.arange(50))
    return dataset


def create_dataset_from_multiple(roots, img_postfix, mask_postfix=None, **kwargs):
    if isinstance(img_postfix, str):
        img_postfix = len(roots) * [img_postfix]
    elif len(img_postfix) == 1:
        img_postfix = len(roots) * list(img_postfix)
    if isinstance(mask_postfix, str) or mask_postfix is None:
        mask_postfix = len(roots) * [mask_postfix]
    elif len(mask_postfix) == 1:
        mask_postfix = len(roots) * list(mask_postfix)
    if not (len(roots) == len(img_postfix) == len(mask_postfix)):
        raise ValueError(
            f"Length mismatch: roots ({len(roots)}), img_postfix ({len(img_postfix)}), "
            f"mask_postfix ({len(mask_postfix)})"
        )
    datasets = []
    for root, img_postfix_, mask_postfix_ in zip(roots, img_postfix, mask_postfix):
        dataset = create_dataset(
            root, img_postfix=img_postfix_, mask_postfix=mask_postfix_, **kwargs
        )
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)


def save_features(
    features_out_dir,
    img_name,
    features_out,
    mask,
):
    name = img_name
    out_path_dir = Path(features_out_dir)
    out_path_dir.mkdir(exist_ok=True, parents=True)

    file_name = f"{name}_feat.png"
    out_path = out_path_dir / file_name
    save_feat_as_img(str(out_path), features_out.clip(-1, 1))


def save_visuals(img_no_norm, img_name, features_out, image_size, vis_out_dir):
    name = img_name
    img_np = img_no_norm.permute(1, 2, 0).cpu().numpy() * 255
    features_out = (
        torch.nn.functional.interpolate(
            torch.from_numpy(features_out[None]).permute(0, 3, 1, 2),
            size=[image_size, image_size],
        )
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .numpy()
    )
    features_out = (features_out + 1) / 2
    features_list = [
        features_out[..., k * 3 : k * 3 + 3] * 255
        for k in range(min(features_out.shape[-1], 9) // 3)
    ]
    visual = [img_np] + features_list
    visual = arrange([visual])

    filename = f"{name}.png"
    out_path = Path(vis_out_dir) / filename
    out_path.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray((visual).astype(np.uint8)).save(out_path)


def extract_features(
    args,
    dataloader,
    extractor,
    pca_mat=None,
    features_out_dir=None,
    vis_out_dir=None,
    log_features=False,
    is_main_process=True,
):
    with torch.no_grad():
        all_features = []
        all_masks = []
        i = 0
        for batch in tqdm(dataloader, total=len(dataloader), disable=not is_main_process):
            unnormalized_image = batch["unnormalized_image"]
            img_name = batch["img_name"]
            mask = batch.get("mask")

            features = extractor.extract(batch)
            b, h, w, c = features.shape
            features = rearrange(features, "b h w c -> (b h w) c")

            features = features.cpu().numpy()
            if args.normalize_features:
                features = features / np.linalg.norm(features, axis=-1, keepdims=True)

            if log_features:
                all_features.append(features)
                if mask is not None:
                    mask_for_log = rearrange(mask, "b c h w -> (b h w) c")
                    all_masks.append(mask_for_log)

            # export image features and save visuals
            if pca_mat is not None:
                features_out = pca_mat.apply_py(features)
            else:
                features_out = features.copy()

            features_out = rearrange(
                features_out, "(b h w) c -> b h w c", b=b, h=h, w=w
            )
            for j, unnormalized_image_ in enumerate(unnormalized_image):
                # save features
                if features_out_dir is not None:
                    current_mask = mask[j] if mask is not None else None
                    save_features(
                        features_out_dir,
                        img_name[j],
                        features_out[j],
                        current_mask,
                    )
                # visuals
                save_visuals_frq = max(
                    int(len(dataloader.dataset) / args.n_images_vis), 1
                )
                if vis_out_dir is not None and i % save_visuals_frq == 0:
                    save_visuals(
                        unnormalized_image_,
                        img_name[j],
                        features_out[j],
                        extractor.image_size,
                        vis_out_dir,
                    )
                i += 1

        if log_features:
            all_features = np.concatenate(all_features, axis=0)
            if all_masks:
                all_masks = np.concatenate(all_masks, axis=0)
            return all_features, all_masks
        return None, None


class FilesListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        files_list,
        img_postfix,
        mask_postfix,
        transform=None,
        mask_transform=None,
    ):
        self.files_list = files_list
        self.img_postfix = img_postfix
        self.mask_root = Path(root_dir) / "masks"
        self.mask_postfix = mask_postfix
        if transform is None:
            transform = {
                "dino_input": transforms.Compose([transforms.ToTensor()]),
                "sd_input": transforms.Compose([transforms.ToTensor()]),
                "unnormalized_image": transforms.Compose([transforms.ToTensor()]),
            }
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_path = self.files_list[idx]
        img_name = img_path.name.removesuffix(self.img_postfix)
        
        try:
            img = Image.open(img_path)
            assert len(img.getbands()) == 3
        except PIL.UnidentifiedImageError as e:
            logger.warning(f"Error: {e}")
            img = Image.new("RGB", (256, 256))
        transformed_img = {}
        for key, transform in self.transform.items():
            transformed_img[key] = transform(img)
            if key == "dino_input":
                assert transformed_img[key].shape == (3, 840, 840)
            else:
                assert transformed_img[key].shape == (3, 960, 960)

        result = {
            **transformed_img,
            "img_path": str(img_path),
            "img_name": img_name,
        }

        if self.mask_postfix is not None:
            mask_path = self.mask_root / (img_name + self.mask_postfix)
            mask = Image.open(mask_path)
            if self.mask_transform is not None:
                mask_transformed = self.mask_transform(mask)[:1]
                result["mask"] = mask_transformed

        return result


@dataclass
class Config:
    test_root: List[str]
    exp_name: str
    results_info_root: str
    features_out_root: Optional[str] = None
    vis_out_root: Optional[str] = None
    load_pca_path: Optional[str] = None
    img_postfix_test: List[str] = field(default_factory=lambda: ["_rgb.png"])
    mask_postfix_test: Optional[List[str]] = None
    batch_size: int = 12
    use_pca: bool = True
    normalize_features: bool = True
    debug_subset: bool = False
    shuffle_data: bool = True
    n_images_vis: int = 100
    test_samples_range: Optional[List[int]] = None


def run(extractor: ExtractorWrapper, args: Config, rank: int = 0, world_size: int = 1):
    # TODO: clean up the args handling
    exp_name = args.exp_name
    batch_size = args.batch_size
    if args.features_out_root is None:
        if len(args.test_root) != 1:
            raise ValueError(
                "`features_out_root` must be specified when `test_root` does not have exactly 1 entry."
            )
        features_out_root = args.test_root[0]
    else:
        features_out_root = args.features_out_root
    features_out_dir = Path(features_out_root) / exp_name

    # first check if we do PCA
    if args.use_pca:
        if args.load_pca_path:
            # check if we were provided with a path PCA matrix
            logger.info(f"Loading PCA from {args.load_pca_path}")
            pca_mat = faiss.read_VectorTransform(args.load_pca_path)
        else:
            raise ValueError("`use_pca` is True but `load_pca_path` is not provided.")
    else:
        pca_mat = None

    logger.info(f"Create dataset")
    dataset = create_dataset_from_multiple(
        args.test_root,
        args.img_postfix_test,
        mask_postfix=args.mask_postfix_test,
        debug_subset=args.debug_subset,
        transform=extractor.transform,
        samples_range=args.test_samples_range,
    )
    ind = list(range(len(dataset)))

    if world_size > 1:
        chunk_size = math.ceil(len(ind) / world_size)
        start = rank * chunk_size
        end = min(start + chunk_size, len(ind))
        local_ind = ind[start:end]

        if len(local_ind) > 0:
            local_dataset = torch.utils.data.Subset(dataset, local_ind)
            dataloader = DataLoader(
                local_dataset,
                batch_size=batch_size,
                shuffle=args.shuffle_data,
                num_workers=2,
                persistent_workers=True,
            )

            logger.info(f"Rank {rank}: Extracting features")
            extract_features(
                args,
                dataloader,
                extractor,
                pca_mat=pca_mat,
                features_out_dir=features_out_dir,
                is_main_process=(rank == 0),
            )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle_data,
            num_workers=2,
            persistent_workers=True,
        )

        logger.info("Extracting features")
        extract_features(
            args,
            dataloader,
            extractor,
            pca_mat=pca_mat,
            features_out_dir=features_out_dir,
            is_main_process=True,
        )

    logger.info("Extracting features done")


if __name__ == "__main__":
    config = parse_config()

    config = OmegaConf.create({"_target_": "__main__.run", **config})
    # Instantiate the function with arguments from the configuration
    hydra.utils.instantiate(config)
