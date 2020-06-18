import numpy as np

def compute_acc(gt,pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    h, w = gt.shape
    for i in range(h):
        for j in range(w):
            if(gt[i][j] == pred[i][j]):
                if(gt[i][j] == 0):
                    TN = TN + 1
                else:
                    TP = TP + 1
            else:
                if(gt[i][j] == 0):
                    FP = FP + 1
                else:
                    FN = FN + 1
    return TN, TP, FP, FN

class Acc_Meter:
    def __init__(self):
        self.reset()
    def update(self,gt,pred):
        l,h,w=gt.shape
        for i in range(l):
            TN, TP, FP, FN=compute_acc(gt[i],pred[i])
            self.N+=h*w
            self.TN+=TN
            self.TP+=TP
            self.FP+=FP
            self.FN+=FN
    def get_acc(self):
        try:prec=self.TP/(self.TP+self.FP)
        except:prec=0
        try:recall=self.TP/(self.TP+self.FN)
        except:recall=0
        try:
            acc=2*prec*recall/(prec+recall)
        except:
            acc=0
        cls_acc=(self.TP+self.TN)/self.N

        return {'prec':prec,
                'recall':recall,
                'acc':acc,
                'cls_acc':cls_acc,
                'TP':self.TP,
                'TN':self.TN,
                'FP':self.FP,
                'FN':self.FN,
                'N':self.N,
                'N_Total':self.TP+self.TN+self.FP+self.FN,
                }
    def reset(self):
        self.N=0
        self.TN=0
        self.TP=0
        self.FP=0
        self.FN=0