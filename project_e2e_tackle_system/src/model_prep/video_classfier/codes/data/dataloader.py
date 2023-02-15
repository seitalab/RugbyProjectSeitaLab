import os
from glob import glob
from typing import Iterable, List, Tuple
from argparse import Namespace

import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

from codes.data.dataset import RugbyVideoDataset
from codes.data.transform_funcs import (
    ToTensor, LoadImage, HorizontalFlip, SaltPepperNoise, 
    ColorJitter
)

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["video_classifier"]

def split_train_val(data: np.ndarray, val_ratio: float):
    """
    Args:

    Returns:

    """
    idx = np.arange(len(data))
    train_idx, valid_idx = train_test_split(
        idx, test_size=val_ratio, random_state=1)
    train, val = data[train_idx], data[valid_idx]
    return train, val

def prepare_dataset() -> Tuple:
    """
    Args:
        None
    Returns:
        datasets (Tuple): Tuple of datasets (train, val, test).
    """

    data_locs = glob(config["data_root"] + "/0*")

    label_csv = os.path.join(config["data_root"], "..", config["manual_label_csv"])
    df_label = pd.read_csv(label_csv).fillna(0)

    dataset = []
    filenames = df_label.loc[:, "filename"].values
    for data_loc in data_locs:
        data_idx = os.path.basename(data_loc)
        if data_idx + ".jpg" in filenames:
            is_target = filenames == data_idx + ".jpg"
            label = df_label.loc[is_target, "label"].values[0]
        else:
            label = 0
        dataset.append((data_loc, label))

    dataset = np.array(dataset)
    trainset, testset = split_train_val(dataset, val_ratio=0.2)
    valset, testset = split_train_val(testset, val_ratio=0.5)

    return (trainset, valset, testset)

def prepare_preprocess(is_train: bool):
    """
    Args:

    Returns:

    """
    transformations = []
    # 
    transformations.append(LoadImage(is_train=is_train))
    if is_train:
        transformations.append(HorizontalFlip())
        transformations.append(ColorJitter())
        transformations.append(SaltPepperNoise())

    # ToTensor and compose.
    transformations.append(ToTensor())
    composed = transforms.Compose(transformations)
    return composed


def prepare_dataloader(args: Namespace) -> Tuple[Iterable, Iterable, Iterable]:
    """
    Args:
        args (Namespace): 
    Returns:
        loaders (Tuple): 
    """

    train_transform = prepare_preprocess(is_train=True)
    test_transform = prepare_preprocess(is_train=False)
    trainset, valset, testset = prepare_dataset()

    trainset = RugbyVideoDataset(trainset, train_transform)
    valset = RugbyVideoDataset(valset, test_transform)
    testset = RugbyVideoDataset(testset, test_transform)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    return trainloader, valloader, testloader