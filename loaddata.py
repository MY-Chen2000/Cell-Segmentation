import os
import glob
import cv2
import numpy as np
import torch

__all__ = ['Dataset', 'ToTensor', 'GrayscaleNormalization', 'RandomFlip', 'RandomRotation']


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, labels_dir, transform=None):
        self.transform = transform
        self.imgs = sorted(glob.glob(os.path.join(imgs_dir, '*.png')))
        self.labels = sorted(glob.glob(os.path.join(labels_dir, '*.png')))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index], cv2.IMREAD_GRAYSCALE) / 255.
        label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE) / 255.

        ret = {
            'img': img[:, :, np.newaxis],
            'label': label[:, :, np.newaxis],
        }

        if self.transform:
            ret = self.transform(ret)

        return ret


class ToTensor:
    def __call__(self, data):
        img, label = data['img'], data['label']

        img = img.transpose(2, 0, 1).astype(np.float32)
        label = label.transpose(2, 0, 1).astype(np.float32)

        ret = {
            'img': torch.from_numpy(img),
            'label': torch.from_numpy(label),
        }
        return ret


class GrayscaleNormalization:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        img, label = data['img'], data['label']
        img = (img - self.mean) / self.std

        ret = {
            'img': img,
            'label': label,
        }
        return ret


class RandomFlip:
    def __call__(self, data):
        img, label = data['img'], data['label']

        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            img = np.flipud(img)
            label = np.flipud(label)

        ret = {
            'img': img,
            'label': label,
        }
        return ret


class RandomRotation:
    def __init__(self, angle=90):
        self.angle = angle

    def __call__(self, data):
        img, label = data['img'], data['label']

        angle = np.random.rand() * self.angle - self.angle/2.0
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1, intborderMode=cv2.BORDER_REFLECT)
        img = cv2.warpAffine(img, M, (cols, rows))
        label = cv2.warpAffine(label, M, (cols, rows))

        ret = {
            'img': img,
            'label': label,
        }
        return ret

