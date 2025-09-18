import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_config():
    # Parse command line arguments - expecting list of configuration files as input argument configs=...
    cli_config = OmegaConf.from_cli()

    # Split the 'configs' argument into a list
    config_files = cli_config.configs.split(",")

    # Remove the 'configs' argument from the command line arguments - not used in the main function
    cli_config.pop("configs")

    # Load and merge configurations from the specified files
    config = OmegaConf.create()
    for filename in config_files:
        file_config = OmegaConf.load(filename)
        config = OmegaConf.merge(config, file_config)

    # Merge command line arguments into the YAML configuration
    # Command line arguments will override YAML configuration if provided
    config = OmegaConf.merge(config, cli_config)

    return config
