from customizedDataset import customizedDataset
from model import MyNet
import matplotlib.pyplot as plt
import numpy as np

import torch
import random

import torch.nn as nn
import os

from utils_sc import *

torch.manual_seed(1)

def visualize_pred(data,pred,visualizerPath,verbose = True,idx = 0):


        # pad blank space if two images dont correspond in dimension2
        typeName = 'src'
        srcImg = data['srcImg'][0].permute(1,2,0)
        srcKps = data['srcKps'][0]
        srcKps_pred = pred
        

        typeName = 'trg'
        trgImg = data['trgImg'][0].permute(1,2,0)
        trgKps = data[typeName+'Kps'][0]
        

        
        if verbose:
            print(srcImg.shape)
            print(trgImg.shape)
            print(srcKps)
        
        n = max(srcImg.shape[1],trgImg.shape[1]) 

        if srcImg.shape[1] > trgImg.shape[1]:
            if data['flip'][0]:
                m = nn.ConstantPad2d((0,0,n - trgImg.shape[1],0),0)
            else:
                m = nn.ConstantPad2d((0,0,0,n - trgImg.shape[1]),0)
            trgImg = m(trgImg)
        else:
            m = nn.ConstantPad2d((0,0,0,n - srcImg.shape[1]),0)
            srcImg = m(srcImg)
        
        
        if verbose:
            print(srcImg.shape)
            print(trgImg.shape)
        

        jointImg = torch.cat((srcImg,trgImg),0)
        
        
        if verbose:
            print(jointImg.shape)
            print(srcKps.shape)
            print(data['flip'][0])
        
        plt.cla()

        
        
        for i in range(srcKps.shape[1]):
            xa = float(srcKps[0,i])
            ya = float(srcKps[1,i])
            xb = float(trgKps[0,i])
            yb = float(trgKps[1,i])+srcImg.shape[0]
            xc = float(srcKps_pred[0,i])
            yc = float(srcKps_pred[1,i])




            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa,ya), radius=5, color=c))
            plt.gca().add_artist(plt.Circle((xb,yb), radius=5, color=c))
            plt.gca().add_artist(plt.Rectangle((xc,yc), width = 5, height = 5, color=c))
            
            plt.plot([xc, xb], [yc, yb], c=c, linestyle='-', linewidth=1.5) 

       
        
     
        plt.imshow(jointImg.cpu().detach().numpy())
        plt.axis('off')

        # remove blank space around the pic
        plt.savefig(visualizerPath+'/debug/'+str(idx)+'_imagePair_pred.jpg',bbox_inches = 'tight',pad_inches = 0.0)




if __name__ == "__main__":
    print('currently executing main.py file')

    visualizerPath = 'visualizer'

    #visualizerPath = ''
    
    trainLoader = torch.utils.data.DataLoader(customizedDataset(visualizerPath = visualizerPath,split='test',verbose = False),batch_size = 1, shuffle = True, num_workers = 0)

    model = MyNet(verbose = False,visualizerPath = visualizerPath)

    pckRes = []
    prev = ''
    for idx, data in enumerate(trainLoader):
        
        print(data['srcName'])
        #print(data['srcImg'].shape)
        #print(data['trgKps'][0])
        #exit()
        inference, src_geometry, trg_geometry = model(data)
        warpedKps = warp_kps(data['trgKps'][0],inference,data['trgImg'][0].shape,verbose = False)
        #warpedKps = predict_kps(src_geometry,trg_geometry,data['srcKps'][0],correlationMap)
        visualize_pred(data,warpedKps,visualizerPath = visualizerPath)
        #print('pck bound: ',data['pckBound'][0])
        res = cal_pck(data['srcKps'],warpedKps,data['pckBound'][0],verbose = False)
        print('pck: ', res,'%')
        exit()
        if(res < 70):
            visualize_pred(data,warpedKps,visualizerPath = 'visualizer',idx = idx,verbose = False)
        pckRes.append(res)
        #print(data['srcName'])
        #exit()
        #prev = data['srcName'][0]
    pckRes = np.array(pckRes)
    print('pck list shape: ',np.array(pckRes).shape)
    print('average pck : ', np.mean(pckRes))