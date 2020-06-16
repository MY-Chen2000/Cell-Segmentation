import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
from loaddata import *
from utils.Unetplus_utils import  *
from utils.UNet_utils import *
from models.unetplus import NestedUNet
from config import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    for batch_idx, data in enumerate(train_loader, 1):
        # Forward Propagation
        input = data['img'].to(device)
        target = data['label'].to(device)

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader, 1):
            # Forward Propagation
            input = data['img'].to(device)
            target = data['label'].to(device)

            # compute output

            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def train_UNetplus():
    cfg = UnetplusConfig()
    criterion = nn.BCEWithLogitsLoss().cuda()
    cudnn.benchmark = True
    model=NestedUNet(1,1,False)
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = optim.SGD(params, lr=cfg.LEARNING_RATE, momentum=cfg.momentum,nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    optimizer = optim.Adam(params, lr=cfg.LEARNING_RATE, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-5)

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

    best_iou = 0
    start_epoch = 0
    num_epochs = cfg.NUM_EPOCHS
    for epoch in range(start_epoch + 1, num_epochs + 1):
        print('Epoch [%d/%d]' % (epoch, num_epochs))
        train_log = train(train_loader, model, criterion, optimizer)
        scheduler.step()
        # evaluate on validation set
        val_log = validate( val_loader, model, criterion)
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        if val_log['iou'] > best_iou:
            os.makedirs('./saved_models/unetplus',exist_ok=True)
            torch.save(model.state_dict(), './saved_models/unetplus/model.pth')
            best_iou = val_log['iou']
            print("=> saved best model")
        torch.cuda.empty_cache()

def test_UNetplus():
    cfg = UnetplusConfig()
    transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    RESULTS_DIR = os.path.join(ROOT_DIR, 'test_results/unetplus')
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

    # Loss Function
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    cudnn.benchmark = True
    # create model
    print("=> creating model unetplus")
    model = NestedUNet(1, 1, False)
    model = model.cuda()
    model.load_state_dict(torch.load('./saved_models/unetplus/model.pth'))
    model.eval()
    with torch.no_grad():
        loss_arr = list()
        for batch_idx, data in enumerate(test_loader, 1):
            # Forward Propagation
            input = data['img'].to(device)
            label = data['label'].to(device)
            output = model(input)

            # Calc Loss Function
            loss = loss_fn(output, label)
            loss_arr.append(loss.item())

            print_form = '[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(batch_idx, test_batch_num, loss_arr[-1]))

            # img = to_numpy(denormalization(img, mean=0.5, std=0.5))
            label = to_numpy(label)
            output = to_numpy(classify_class(output))

            for j in range(label.shape[0]):
                crt_id = int(test_batch_num * (batch_idx - 1) + j)
                # plt.imsave(os.path.join(RESULTS_DIR, f'img/{crt_id:04}.png'), img[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(label_save_path, f'{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(output_save_path, f'{crt_id:04}.png'), output[j].squeeze(), cmap='gray')

            # print_form = '[Result] | Avg Loss: {:0.4f}'
            # print(print_form.format(np.mean(loss_arr)))
        unet_acc(output_save_path, label_save_path)


