import numpy as np
from torch.utils.data import Dataset,DataLoader
import cv2
from torchvision import transforms
import torch
#file='./dataset/dataset/new train set/train_label/0.png.npy'
#train_label=np.load(file)
#print(train_label[0])

class SData(Dataset):
    def __init__(self,args,split,transform=None):
        if split=='train':
            self.img_root_path='./dataset/dataset/new train set/train_img/'
            self.lbl_root_path='./dataset/dataset/new train set/train_label/'
        else:
            self.img_root_path = './dataset/dataset/new_test_set/test_img/'
            self.lbl_root_path = './dataset/dataset/new_test_set/test_label/'
        self.is_train=split=='train'
        self.nfiles=25 if self.is_train else 5
        self.load_data()
        self.transform=transform
    def load_data(self):
        self.data=[]
        self.lbl=[]
        for i in range(self.nfiles):
            img_name=self.img_root_path+'{}.png'.format(i)
            label_name=self.lbl_root_path+'{}.png'.format(i)
            #img=cv2.imread(img_name)[:,:,:1].transpose(2,0,1)
            img=(cv2.imread(img_name)/255)# [:,:,np.newaxis]
            train_label=(cv2.imread(label_name,cv2.IMREAD_GRAYSCALE)/255)#[:,:,np.newaxis]
            # print(img.shape,train_label.shape)
            self.data.append(img)
            self.lbl.append(train_label)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            rr=self.transform({'data':self.data[index], 'lbl':self.lbl[index]})
            return rr['data'],rr['lbl']
        return self.data[index], self.lbl[index]

class GrayscaleNormalization:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, rr):
        img,label=rr['data'],rr['lbl']
        img = (img - self.mean) / self.std
        return {'data':img,'lbl':label}


class RandomFlip:
    def __call__(self, rr):
        img, label = rr['data'], rr['lbl']
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            img = np.flipud(img)
            label = np.flipud(label)
        return {'data':img,'lbl':label}

class ToTensor:
    def __call__(self, rr):
        img, label = rr['data'], rr['lbl']
        img = img.transpose(2, 0, 1).astype(np.float32)
        label = label.astype(np.int)

        return {'data':img,'lbl':label}

def get_loaders(args):
    train_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        RandomFlip(),
        ToTensor(),
    ])
    val_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])
    train_ds=SData(args,'train',transform=train_transform)
    valid_ds=SData(args,'valid',transform=val_transform)
    train_loader=DataLoader(train_ds,args.batch_size,shuffle=True,drop_last=False)
    valid_loader=DataLoader(valid_ds,args.batch_size,shuffle=False,drop_last=False)
    return train_loader,valid_loader