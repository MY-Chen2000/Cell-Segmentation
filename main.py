import argparse
from Unet_run import train_UNet, test_UNet
from Unetplus_run import train_UNetplus, test_UNetplus
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='UNet', type=str, help='model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()

    print(args.model_name)
    #print(args.test)
    if args.model_name=='UNet':
        if args.test==False:
            train_UNet()
        else:
            test_UNet()

    if args.model_name=='UNetplus':
        if args.test==False:
            train_UNetplus()
        else:
            test_UNetplus()