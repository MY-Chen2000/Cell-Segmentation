import argparse
from models.fcnet import FCNet
from models.cenet import CE_Net_
import torch
import time
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR
# from utloss import cross_entropy2d
from load_data import get_loaders
# from metrics import runningScore,averageMeter
# from my_acc import Acc_Meter
from tqdm import tqdm
import cv2
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='UNet', type=str, help='model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    #parser.add_argument('--gpus', type=str, default='0',help='gpu ids')
    args = parser.parse_args()
    args.n_classes=2
    args.batch_size=8
    if args.model_name == 'FCNet':
        model = FCNet(args).cuda()
        model = torch.nn.DataParallel(model)
    elif args.model_name == 'CENet':
        model = CE_Net_(args).cuda()
        model = torch.nn.DataParallel(model)
    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint["model_state"])
        else:
            print('Unable to load {}'.format(args.model_name))

    train_loader, valid_loader=get_loaders(args)

    model.eval()
    try:
        os.mkdir('preds/')
    except:
        pass
    try:
        os.mkdir('preds/model_name')
    except:
        pass
    i=0
    with torch.no_grad():
        for i_val, (images_val, labels_val) in tqdm(enumerate(valid_loader)):
            images_val = images_val.cuda()#to(device)
            labels_val = labels_val.cuda()#to(device)

            outputs = model(images_val)

            pred = outputs.argmax(1)
            # print(pred)
            pred=pred.detach().cpu().numpy()
            for pp in pred:
                cv2.imwrite('preds/model_name/{}.png'.format(i),pp*255)
                i+=1