import os
import argparse
from logging import getLogger
import src.resnet50 as resnet_models
import torch
from pathlib import Path

logger = getLogger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    args = parser.parse_args()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))
    else:
        raise ValueError("No pretrained weights found")

    new_path = Path(args.pretrained)
    new_path = new_path.parent / f"{new_path.name}.oldformat"

    torch.save(state_dict, str(new_path), _use_new_zipfile_serialization=False)
