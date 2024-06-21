from pathlib import Path
import yaml

# Function to read a configuration file in YAML format
def readConfig(cfg_file_path: Path) -> dict:
    """Read config file

    Args:
        cfg_file_path (Path): path to config file

    Returns:
        dict: config file data
    """
    # Open and load the YAML configuration file
    config = yaml.load(open(cfg_file_path, "r"), Loader=yaml.FullLoader)
    return config

