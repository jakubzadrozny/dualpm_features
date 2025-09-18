import numpy as np
import torch
from PIL import Image


def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize(
                    (
                        target_res,
                        int(np.around(target_res * original_height / original_width)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2 : (width + height) // 2] = img
        else:
            if resize:
                img = img.resize(
                    (
                        int(np.around(target_res * original_width / original_height)),
                        target_res,
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2 : (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize(
                    (
                        target_res,
                        int(np.around(target_res * original_height / original_width)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(
                img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode="edge"
            )
        else:
            if resize:
                img = img.resize(
                    (
                        int(np.around(target_res * original_width / original_height)),
                        target_res,
                    ),
                    Image.Resampling.LANCZOS,
                )
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(
                img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode="edge"
            )
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas
