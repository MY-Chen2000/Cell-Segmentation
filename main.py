import argparse
from Unet import train_Unet, test_Unet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='UNet', type=str, help='model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()

    print(args.model_name)
    print(args.test)
    if args.model_name=='UNet':
        if args.test==False:
            train_Unet()
        else:
            test_Unet(args.model_path)