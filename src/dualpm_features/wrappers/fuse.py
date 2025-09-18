import torch
from einops import rearrange
from torchvision import transforms

from dualpm_features.third_party.fuse.extractor_dino import ViTExtractor
from dualpm_features.third_party.fuse.extractor_sd import (
    load_model,
    process_features_and_mask,
)
from dualpm_features.third_party.fuse.utils import resize
from dualpm_features.wrappers.base import ExtractorWrapper


class FuseWrapper(ExtractorWrapper):
    def __init__(
        self,
        device="cuda:0",
    ):
        super().__init__(device)
        self.normalize = True

        self.fuse_dino = True

        self.sd_image_size = 960
        self.dino_image_size = 840

        self.image_size = self.sd_image_size

        self.dino_extractor = ViTExtractor(device=self.device)

        self.sd_model, self.sd_augmentation = load_model()

        dino_patch_size = self.dino_extractor.model.patch_embed.patch_size[0]
        dino_stride = self.dino_extractor.model.patch_embed.proj.stride[0]

        self.dino_num_patches = int(
            dino_patch_size
            / dino_stride
            * (self.dino_image_size // dino_patch_size - 1)
            + 1
        )

        sd_image_transform = transforms.Compose(
            [
                lambda x: resize(x, self.sd_image_size, resize=True, to_pil=True),
                transforms.ToTensor(),
            ]
        )
        dino_image_transform = transforms.Compose(
            [
                lambda x: resize(x, self.dino_image_size, resize=True, to_pil=True),
                transforms.ToTensor(),
            ]
        )
        self.transform = {
            "dino_input": dino_image_transform,
            "sd_input": sd_image_transform,
        }
        self.transform["unnormalized_image"] = transforms.Compose(
            [
                transforms.Resize(
                    (self.sd_image_size, self.sd_image_size), interpolation=3
                ),
                transforms.ToTensor(),
            ]
        )

    def extract_single_img(self, dino_input_pil, sd_input_pil):
        with torch.no_grad():
            sd_features = process_features_and_mask(
                self.sd_model,
                self.sd_augmentation,
                sd_input_pil,
            )
            sd_features = rearrange(sd_features, "b c h w -> b 1 (h w) c")
            dino_input_tensor = self.dino_extractor.preprocess_pil(dino_input_pil)
            dino_features = self.dino_extractor.extract_descriptors(
                dino_input_tensor.to(self.device),
            )

            if self.normalize:
                # normalize the features
                sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)
                dino_features = dino_features / dino_features.norm(dim=-1, keepdim=True)

            # cat two features together
            combined_features = torch.cat((sd_features, dino_features), dim=-1)

        return combined_features

    def extract(self, batch):
        # TODO make it batched
        batch_features = []
        for i in range(len(batch["dino_input"])):
            dino_input_tensor = batch["dino_input"][i]
            sd_input_tensor = batch["sd_input"][i]
            # convert to PIL
            dino_input_pil = transforms.ToPILImage()(dino_input_tensor)
            sd_input_pil = transforms.ToPILImage()(sd_input_tensor)
            combined_features = self.extract_single_img(dino_input_pil, sd_input_pil)
            batch_features.append(combined_features)
        batch_combined_features = torch.cat(batch_features, dim=0)

        # TODO: support non-square images?
        feature_map_width = int(batch_combined_features.shape[2] ** 0.5)
        feature_map_height = feature_map_width
        reshaped_features = rearrange(
            batch_combined_features,
            "b 1 (h  w) c -> b h w c",
            h=feature_map_height,
            w=feature_map_width,
        )

        return reshaped_features
