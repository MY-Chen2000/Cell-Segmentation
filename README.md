SJTU CS420 Final Project
================
Dataset
----------------
ISBI Challenge: Segmentation of neuronal structures in EM stacks
http://brainiac2.mit.edu/isbi_challenge/

Results
-------------------
* Unet 92.02
* UNet++ 92.17
* FCNet++ 88.66
* CENet++ 91.51

Requirements
-------------
* python3.7
* pytorch>1.3


Usage
-------------
Unet train and test

    python main.py
    python main.py --test
UNet++ train and test
        
    python main.py --model_name UNetplus
    python main.py --model_name UNetplus --test
FCN train and test
        
    python main.py --model_name FCNet
    python main.py --model_name FCNet --test
CENet train and test
        
    python main.py --model_name CENet
    python main.py --model_name CENet --test
