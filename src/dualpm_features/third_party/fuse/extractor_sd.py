"""
From https://github.com/Junyi42/sd-dino/tree/master
"""

import itertools
from contextlib import ExitStack

# from utils.utils_correspondence import resize
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import LazyCall as L
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, random_color
from einops import rearrange
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference
from PIL import Image

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(
        itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES)
    )
)


class StableDiffusionSeg:
    def __init__(self, model, aug):
        """
        Args:
            model (nn.Module):
            aug:
        """
        self.model = model
        self.aug = aug
        self.cpu_device = torch.device("cpu")

    def get_features(self, original_image, pca=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            features (dict):
                the output of the model for one image only.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        features = self.model.get_features([inputs], pca=pca)
        return features


def build_demo_classes_and_metadata():
    demo_thing_classes = COCO_THING_CLASSES
    demo_stuff_classes = COCO_STUFF_CLASSES
    demo_thing_colors = COCO_THING_COLORS
    demo_stuff_colors = COCO_STUFF_COLORS

    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata


import sys


def load_model(
    config_path="Panoptic/odise_label_coco_50e.py",
    seed=42,
    diffusion_ver="v1-5",
    image_size=960,
    num_timesteps=100,
):
    cfg = model_zoo.get_config(config_path, trained=True)

    cfg.model.backbone.feature_extractor.init_checkpoint = "sd://" + diffusion_ver
    cfg.model.backbone.feature_extractor.steps = (num_timesteps,)
    cfg.model.backbone.feature_extractor.unet_block_indices = (2, 5, 8, 11)
    cfg.model.backbone.feature_extractor.encoder_only = False
    cfg.model.backbone.feature_extractor.decoder_only = True
    cfg.model.backbone.feature_extractor.resblock_only = False
    cfg.model.overlap_threshold = 0
    seed_all_rng(seed)

    cfg.dataloader.test.mapper.augmentations = [
        L(T.ResizeShortestEdge)(
            short_edge_length=image_size, sample_style="choice", max_size=2560
        ),
    ]
    dataset_cfg = cfg.dataloader.test

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)

    return model, aug


def get_features(model, aug, image, pca=False):
    demo_classes, demo_metadata = build_demo_classes_and_metadata()
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = StableDiffusionSeg(inference_model, aug)
        features = demo.get_features(np.array(image), pca=pca)
        return features


def pca_process(features):
    # Get the feature tensors
    s5 = features["s5"]
    s4 = features["s4"]
    s3 = features["s3"]

    # Define the target dimensions
    target_dims = {"s5": 128, "s4": 128, "s3": 128}

    # Apply PCA to each tensor using Faiss CPU
    for name, tensor in zip(["s5", "s4", "s3"], [s5, s4, s3]):
        target_dim = target_dims[name]

        # Reshape for PCA
        tensor_rearranged = rearrange(tensor, "b c h w -> b (h w) c")

        # Initialize a Faiss PCA object
        pca = faiss.PCAMatrix(tensor_rearranged.shape[-1], target_dim)

        # Train the PCA object
        pca.train(tensor_rearranged[0].cpu().numpy())

        # Apply PCA to the data
        transformed_tensor_np = pca.apply(tensor_rearranged[0].cpu().numpy())

        # Convert the transformed data back to a tensor
        transformed_tensor = torch.tensor(
            transformed_tensor_np, device=tensor.device
        ).unsqueeze(0)

        # Store the transformed tensor in the features dictionary
        features[name] = transformed_tensor

    # Reshape the tensors back to their original shapes
    features["s5"] = rearrange(
        features["s5"], "b (h w) c -> b c h w", h=s5.shape[-2], w=s5.shape[-1]
    )
    features["s4"] = rearrange(
        features["s4"], "b (h w) c -> b c h w", h=s4.shape[-2], w=s4.shape[-1]
    )
    features["s3"] = rearrange(
        features["s3"], "b (h w) c -> b c h w", h=s3.shape[-2], w=s3.shape[-1]
    )
    # Upsample s5 spatially by a factor of 2
    upsampled_s5 = torch.nn.functional.interpolate(
        features["s5"], scale_factor=2, mode="bilinear", align_corners=False
    )

    # Concatenate upsampled_s5 and s4 to create a new s5
    features["s5"] = torch.cat((upsampled_s5, features["s4"]), dim=1)

    # Set s3 as the new s4
    features["s4"] = features["s3"]

    # Remove s3 from the features dictionary
    del features["s3"]

    return features


def process_features_and_mask(model, aug, image, mask=False, pca=False, raw=False):
    features = get_features(model, aug, image, pca=(pca or raw))
    if pca:
        features = pca_process(features)
    if raw:
        return features
    features_gether_s4_s5 = torch.cat(
        [
            features["s4"],
            F.interpolate(
                features["s5"], size=(features["s4"].shape[-2:]), mode="bilinear"
            ),
        ],
        dim=1,
    )

    if mask:
        raise NotImplementedError(
            "Masking is not supported in this refactored version."
        )

    return features_gether_s4_s5


if __name__ == "__main__":
    image_path = sys.argv[1]
    try:
        input_text = sys.argv[2]
    except:
        input_text = None

    model, aug = load_model()
    img_size = 960
    image = Image.open(image_path).convert("RGB")
    # Assuming resize function is available or needs to be added
    # For now, we'll just resize the image directly
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)

    features = process_features_and_mask(
        model, aug, image, mask=False, pca=False, raw=True
    )
    features = features["s4"]  # save the features of layer 5

    # save the features
    np.save(image_path[:-4] + ".npy", features.cpu().numpy())
