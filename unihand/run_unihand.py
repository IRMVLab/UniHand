import os
import sys
sys.path.append('.')
from data_utils.utils import load_config


if __name__ == "__main__":

    config = load_config(config_path="configs/traineval.yaml")
    gpu_devices = config["gpu"]["visible_devices"]
    # Final command
    COMMANDLINE = f"CUDA_VISIBLE_DEVICES={gpu_devices} python traineval.py"

    print("Running command: " + COMMANDLINE)
    os.system(COMMANDLINE)
