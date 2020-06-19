import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt

from loaddata import *
from models.unet import UNet
from utils.UNet_utils import *
from config import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_UNet():
    cfg = UnetConfig()
    train_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        RandomRotation(),
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

    # Network
    net = UNet().to(device)
    print(count_parameters(net))
    # Loss Function
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    # Optimizer
    optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

    # Tensorboard
    # train_writer = SummaryWriter(log_dir=TRAIN_LOG_DIR)
    # val_writer = SummaryWriter(log_dir=VAL_LOG_DIR)

    # Training
    start_epoch = 0
    # Load Checkpoint File
    if os.listdir(os.path.join(CKPT_DIR,'unet')):
        net, optim, start_epoch = load_net(ckpt_dir=os.path.join(CKPT_DIR,'unet'), net=net, optim=optim)
    else:
        print('* Training from scratch')

    num_epochs = cfg.NUM_EPOCHS
    for epoch in range(start_epoch + 1, num_epochs + 1):
        net.train()
        train_loss_arr = list()

        for batch_idx, data in enumerate(train_loader, 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)

            output = net(img)

            # Backward Propagation
            optim.zero_grad()

            loss = loss_fn(output, label)
            loss.backward()

            optim.step()

            # Calc Loss Function
            train_loss_arr.append(loss.item())
            print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(epoch, num_epochs, batch_idx, train_batch_num, train_loss_arr[-1]))


        train_loss_avg = np.mean(train_loss_arr)
        # train_writer.add_scalar(tag='loss', scalar_value=train_loss_avg, global_step=epoch)

        # Validation (No Back Propagation)
        with torch.no_grad():
            net.eval()  # Evaluation Mode
            val_loss_arr = list()

            for batch_idx, data in enumerate(val_loader, 1):
                # Forward Propagation
                img = data['img'].to(device)
                label = data['label'].to(device)

                output = net(img)

                # Calc Loss Function
                loss = loss_fn(output, label)
                val_loss_arr.append(loss.item())

                print_form = '[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
                print(print_form.format(epoch, num_epochs, batch_idx, val_batch_num, val_loss_arr[-1]))


        val_loss_avg = np.mean(val_loss_arr)
        # val_writer.add_scalar(tag='loss', scalar_value=val_loss_avg, global_step=epoch)

        print_form = '[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f}'
        print(print_form.format(epoch, train_loss_avg, val_loss_avg))
        if epoch % 10 == 0:
            save_net(ckpt_dir=os.path.join(CKPT_DIR,'unet'), net=net, optim=optim, epoch=epoch)

    # train_writer.close()
    # val_writer.close()


def test_UNet():
    cfg = UnetConfig()
    transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    RESULTS_DIR = os.path.join(ROOT_DIR, 'test_results/unet')
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    label_save_path = os.path.join(RESULTS_DIR, 'label')
    output_save_path = os.path.join(RESULTS_DIR, 'output')
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path, exist_ok=True)
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path, exist_ok=True)

    test_dataset = Dataset(imgs_dir=TEST_IMGS_DIR, labels_dir=TEST_LABELS_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    test_data_num = len(test_dataset)
    test_batch_num = int(np.ceil(test_data_num / cfg.BATCH_SIZE))

    # Network
    net = UNet().to(device)

    # Loss Function
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    # Optimizer
    optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

    start_epoch = 0

    # Load Checkpoint File
    if os.listdir(CKPT_DIR):
        net, optim, _ = load_net(ckpt_dir=os.path.join(CKPT_DIR,'unet'), net=net, optim=optim)

    # Evaluation
    with torch.no_grad():
        net.eval()
        loss_arr = list()

        for batch_idx, data in enumerate(test_loader, 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)

            output = net(img)

            # Calc Loss Function
            loss = loss_fn(output, label)
            loss_arr.append(loss.item())

            print_form = '[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(batch_idx, test_batch_num, loss_arr[-1]))

            label = to_numpy(label)
            output = to_numpy(classify_class(output))

            for j in range(label.shape[0]):
                crt_id = int(test_batch_num * (batch_idx - 1) + j)
                plt.imsave(os.path.join(label_save_path, f'{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(output_save_path, f'{crt_id:04}.png'), output[j].squeeze(), cmap='gray')

    unet_acc(output_save_path, label_save_path)


