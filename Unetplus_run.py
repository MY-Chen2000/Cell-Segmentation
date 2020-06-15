import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from loaddata import *
from utils.Unetplus_utils import  *
from config import *

def train_UNetplus():
    cfg = UnetConfig()
    train_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        RandomFlip(),
        ToTensor(),
    ])
    val_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    # Set Dataset
    train_dataset = Dataset(imgs_dir=TRAIN_IMGS_DIR, labels_dir=TRAIN_LABELS_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataset = Dataset(imgs_dir=VAL_IMGS_DIR, labels_dir=VAL_LABELS_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    train_data_num = len(train_dataset)
    val_data_num = len(val_dataset)

    train_batch_num = int(np.ceil(train_data_num / cfg.BATCH_SIZE))  # np.ceil
    val_batch_num = int(np.ceil(val_data_num / cfg.BATCH_SIZE))

    start_epoch = 0
    num_epochs = cfg.NUM_EPOCHS
    for epoch in range(start_epoch + 1, num_epochs + 1):
        for batch_idx, data in enumerate(train_loader, 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)


def test_UNetplus():
    pass