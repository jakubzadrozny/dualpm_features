"""
From https://github.com/Junyi42/sd-dino/tree/master
"""

import argparse
import math
import types
from pathlib import Path
from typing import List, Tuple, Union

import timm
import torch
import torch.nn.modules.utils as nn_utils
import torchvision.transforms
from einops import rearrange
from PIL import Image
from torch import nn
from torchvision import transforms
from typing_extensions import Concatenate, Self, TypeGuard


class ViTExtractor:
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.
    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self,
        model_type: str = "dinov2_vitb14",
        stride: int = 14,
        model: nn.Module = None,
        device: str = "cuda",
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        if type(self.p) == tuple:
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (
            (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )
        self.std = (
            (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        if "v2" in model_type:
            model = torch.hub.load(
                "facebookresearch/dinov2:85a24602099d397264d5b30461ad7f3bfd726ca1",
                model_type,
            )
        elif "dino" in model_type:
            model = torch.hub.load(
                "facebookresearch/dino:main",
                model_type,
            )
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                "vit_small_patch16_224": "dino_vits16",
                "vit_small_patch8_224": "dino_vits8",
                "vit_base_patch16_224": "dino_vitb16",
                "vit_base_patch8_224": "dino_vitb8",
            }
            model = torch.hub.load(
                "facebookresearch/dino:main", model_type_dict[model_type]
            )
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict["head.weight"]
            del temp_state_dict["head.bias"]
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                rearrange(
                    patch_pos_embed,
                    "b (h w) c -> b c h w",
                    h=int(math.sqrt(N)),
                    w=int(math.sqrt(N)),
                ),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = rearrange(patch_pos_embed, "b c h w -> b (h w) c")
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in stride]), (
            f"stride {stride} should divide patch_size {patch_size}"
        )

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride), model
        )
        return model

    def preprocess(
        self,
        image_path: Union[str, Path],
        load_size: Union[int, Tuple[int, int]] = None,
        patch_size: int = 14,
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """

        def divisible_by_num(num, dim):
            return num * (dim // num)

        pil_image = Image.open(image_path).convert("RGB")
        if load_size is not None:
            pil_image = transforms.Resize(
                load_size, interpolation=transforms.InterpolationMode.LANCZOS
            )(pil_image)

            width, height = pil_image.size
            new_width = divisible_by_num(patch_size, width)
            new_height = divisible_by_num(patch_size, height)
            pil_image = pil_image.resize(
                (new_width, new_height), resample=Image.LANCZOS
            )

        prep = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def preprocess_pil(self, pil_image):
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        prep = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )
        prep_img = prep(pil_image)[None, ...]
        return prep_img

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input)
            qkv = rearrange(
                qkv,
                "b n (h d) -> b n h d",
                h=3,
                d=C // module.num_heads,
            )
            qkv = rearrange(qkv, "b n h (d c) -> h b n d c", c=module.num_heads)
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self, batch: torch.Tensor, layers: List[int] = 11, facet: str = "key"
    ) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = rearrange(x, "b h (w1 w2) d -> b (h d) w1 w2", w1=self.num_patches[0])

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3**k
            avg_pool = torch.nn.AvgPool2d(
                win_size, stride=1, padding=win_size // 2, count_include_pad=False
            )
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros(
            (B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])
        ).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3**k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(
                            x - kernel_size, x + kernel_size + 1, kernel_size
                        ):
                            if i == y and j == x and k != 0:
                                continue
                            if (
                                0 <= i < self.num_patches[0]
                                and 0 <= j < self.num_patches[1]
                            ):
                                bin_x[
                                    :,
                                    part_idx * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[
                                    :,
                                    part_idx * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, temp_i, temp_j]
                            part_idx += 1
        bin_x = rearrange(bin_x, "b c h w -> b 1 (h w) c")
        return bin_x

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        layer: int = 11,
        facet: str = "token",
        bin: bool = False,
        include_cls: bool = False,
    ) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == "token":
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, (
                "bin = True and include_cls = True are not supported together, set one of them False."
            )
        if not bin:
            desc = rearrange(x, "b h t d -> b 1 t (h d)")
        else:
            desc = self._log_bin(x)
        return desc
