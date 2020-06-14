import numpy as np

file='../dataset/new train set/train_label/0.png.npy'
train_label=np.load(file)
print(train_label[0])