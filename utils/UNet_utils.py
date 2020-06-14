import os
import torch
from PIL import Image
import numpy as np


__all__ = ['to_numpy', 'denormalization', 'classify_class', 'save_net', 'load_net', 'unet_acc']


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # (Batch, H, W, C)

def denormalization(data, mean, std):
    return (data * std) + mean

def classify_class(x):
    return 1.0 * (x > 0.5)

def save_net(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save(
        {'net': net.state_dict(),'optim': optim.state_dict()},
        os.path.join(ckpt_dir, f'model_epoch{epoch:04}.pth'),
    )
    
def load_net(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda fname: int(''.join(filter(str.isdigit, fname))))
    
    ckpt_path = os.path.join(ckpt_dir, ckpt_list[-1])
    if torch.cuda.is_available():
        model_dict = torch.load(ckpt_path)
    else:
        model_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    print(f'* Load {ckpt_path}')

    net.load_state_dict(model_dict['net'])
    optim.load_state_dict(model_dict['optim'])
    epoch = int(''.join(filter(str.isdigit, ckpt_list[-1])))
    
    return net, optim, epoch

def unet_acc_single(path_output, path_label):
    img = Image.open(path_output)
    # img.show()
    label = Image.open(path_label)
    img = np.array(img).astype(int)
    label = np.array(label).astype(int)
    # print(label[250])
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # print(img.shape)
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            if (img[i][j][0] == label[i][j][0]):
                if (label[i][j][0] == 0):
                    TN = TN + 1
                else:
                    TP = TP + 1
            else:
                if (label[i][j][0] == 0):
                    FP = FP + 1
                else:
                    FN = FN + 1
    return TN, TP, FP, FN

def unet_acc(output_save_path, label_save_path):
    all_TN = 0
    all_TP = 0
    all_FP = 0
    all_FN = 0

    for name in os.listdir(output_save_path):
        img_file = os.path.join(output_save_path, name)
        print(img_file)
        label_file = os.path.join(label_save_path, name)
        TN, TP, FP, FN = unet_acc_single(img_file, label_file)
        all_TN = all_TN + TN
        all_TP = all_TP + TP
        all_FP = all_FP + FP
        all_FN = all_FN + FN
    acc = (all_TP + all_TN) / (all_TP + all_FN + all_FP + all_TN)

    print('TP', all_TP)
    print('FP', all_FP)
    print('FN', all_FN)
    print('TN', all_TN)

    print('accuracy', acc)
