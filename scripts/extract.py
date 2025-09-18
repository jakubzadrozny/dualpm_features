from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from dualpm_features.utils import parse_config

if __name__ == "__main__":
    config = parse_config()

    config = OmegaConf.create({"_target_": "dualpm_features.extract.run", **config})
    # Instantiate the function with arguments from the configuration
    hydra.utils.instantiate(config)
