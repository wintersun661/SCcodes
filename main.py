from customizedDataset import customizedDataset
from model import MyNet
import matplotlib.pyplot as plt
import numpy as np

import torch
import random

import os

torch.manual_seed(1)



if __name__ == "__main__":
    print('currently executing main.py file')

    visualizerPath = 'visualizer'
    
    trainLoader = torch.utils.data.DataLoader(customizedDataset(visualizerPath = visualizerPath),batch_size = 1, shuffle = True, num_workers = 0)

    model = MyNet(verbose = True,visualizerPath = visualizerPath)

    for idx, data in enumerate(trainLoader):
        print(data['srcImg'].shape)
        print(data['srcImg'].permute(2,0,1).shape)
        features = model(data)
        print(data['srcName'])
        
        

        exit()