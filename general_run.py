import argparse
from models.fcnet import FCNet
from models.cenet import CE_Net_
import torch
import time
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR
from utils.loss import cross_entropy2d,DiceLoss
from load_data import get_loaders
from utils.metrics import runningScore,averageMeter
from tqdm import tqdm
from utils.my_acc import Acc_Meter
from tensorboardX import SummaryWriter
import os

def train_general(args):
    args.optimizer='Adam'
    args.n_classes = 2
    args.batch_size = 8
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # print(args.model_name)
    # print(args.test)
    if args.model_name == 'FCNet':
        model = FCNet(args).cuda()
        model = torch.nn.DataParallel(model)
        if args.optimizer == 'SGD':
            optimizer = SGD(model.parameters(), .1, weight_decay=5e-4, momentum=.99)
        elif args.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), .1, weight_decay=5e-4)
        criterion = cross_entropy2d
        scheduler = MultiStepLR(optimizer, [100, 200, 400, 800, 3200], .1)
    elif args.model_name == 'CENet':
        model = CE_Net_(args).cuda()
        model = torch.nn.DataParallel(model)
        if args.optimizer == 'SGD':
            optimizer = SGD(model.parameters(), .1, weight_decay=5e-4, momentum=.99)
            scheduler = MultiStepLR(optimizer, [100, 200, 400, 800, 3200], .1)
        elif args.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), .001, weight_decay=5e-4)
            scheduler = MultiStepLR(optimizer, [400, 3200], .1)
        # criterion = cross_entropy2d
        criterion = DiceLoss()
        # scheduler = MultiStepLR(optimizer, [100, 200, 400, 800, 3200], .1)
    start_iter = 0
    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
        else:
            print('Unable to load {}'.format(args.model_name))

    train_loader, valid_loader = get_loaders(args)

    try:
        os.mkdir('logs/')
    except:
        pass
    try:
        os.mkdir('results/')
    except:
        pass
    try:
        os.mkdir('results/' + args.model_name)
    except:
        pass
    writer = SummaryWriter(log_dir='logs/')

    best = -100.0
    i = start_iter
    flag = True

    running_metrics_val = Acc_Meter()
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    # while i <= args.niter and flag:
    while i <= 300000 and flag:
        for (images, labels) in train_loader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            # if (i + 1) % cfg["training"]["print_interval"] == 0:
            if (i + 1) % 50 == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    300000,
                    loss.item(),
                    time_meter.avg / args.batch_size,
                )

                print(print_str)
                # logger.info(print_str)
                # writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                # time_meter.reset()

            # if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
            if (i + 1) % 500 == 0 or (i + 1) == 300000:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valid_loader)):
                        images_val = images_val.cuda()  # to(device)
                        labels_val = labels_val.cuda()  # to(device)

                        outputs = model(images_val)
                        # val_loss = loss_fn(input=outputs, target=labels_val)
                        val_loss = criterion(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                # writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                print("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                results = running_metrics_val.get_acc()
                for k, v in results.items():
                    writer.add_scalar(k, v, i + 1)
                print(results)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if results['cls_acc'] >= best:
                    best = results['cls_acc']
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best": best,
                    }
                    save_path = os.path.join(
                        "results/{}/results_{}_best_model.pkl".format(args.model_name, i + 1),
                    )
                    torch.save(state, save_path)

            if (i + 1) == 300000:
                flag = False
                break
    writer.close()

def valid_general(args):
    args.n_classes = 2
    args.batch_size = 8
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # print(args.model_name)
    # print(args.test)
    if args.model_name == 'FCNet':
        model = FCNet(args).cuda()
        model = torch.nn.DataParallel(model)
        criterion = cross_entropy2d
    elif args.model_name == 'CENet':
        model = CE_Net_(args).cuda()
        model = torch.nn.DataParallel(model)
        criterion = cross_entropy2d
    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint["model_state"])
        else:
            print('Unable to load {}'.format(args.model_name))

    train_loader, valid_loader = get_loaders(args)

    best_iou = -100.0
    i = start_iter
    flag = True

    # running_metrics_val = runningScore(args.n_classes)
    running_metrics_val = Acc_Meter()
    val_loss_meter = averageMeter()
    time_meter = averageMeter()
    model.eval()
    with torch.no_grad():
        for i_val, (images_val, labels_val) in tqdm(enumerate(valid_loader)):
            images_val = images_val.cuda()  # to(device)
            labels_val = labels_val.cuda()  # to(device)

            outputs = model(images_val)
            # val_loss = loss_fn(input=outputs, target=labels_val)
            val_loss = criterion(input=outputs, target=labels_val)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics_val.update(gt, pred)
            val_loss_meter.update(val_loss.item())

    # writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
    print("Loss: %.4f" % (val_loss_meter.avg))

    # score, class_iou = running_metrics_val.get_scores()
    results = running_metrics_val.get_acc()
    print(results)
    '''for k, v in score.items():
        print(k, v)
        print("{}: {}".format(k, v))
        #writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

    for k, v in class_iou.items():
        print("{}: {}".format(k, v))
        #writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)'''

    val_loss_meter.reset()
    running_metrics_val.reset()