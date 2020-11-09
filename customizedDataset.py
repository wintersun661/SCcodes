import os

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import scipy.io as sio
import numpy as np


class customizedDataset(Dataset):

    # benchmark: foldername, imgSubFolderName, annoSubFolderName,classNameList
    metaData = {'pascal':\
                    ['PF-dataset-PASCAL','JPEGImages','Annotations',
                    ['placeholder','aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']],}

    def __init__(self,benchmark = 'pascal', datapath = 'dataset', split = 'train',verbose = True, transform = None, device = 'cuda:0',pckType = 'img'):
        
        self.benchmark = benchmark
        self.pckType = pckType
        self.splitType = split
        self.verbose = verbose
        self.transform = transform
        self.device = device
        
        self.basePath = os.path.join(os.path.abspath(datapath),self.metaData[self.benchmark][0])
        self.imgPath = os.path.join(self.basePath,self.metaData[self.benchmark][1])
        self.annoPath = os.path.join(self.basePath,self.metaData[self.benchmark][2])

        self.tmpPath = os.path.join(self.basePath,'tmp')

        self.classNameList = self.metaData[self.benchmark][3]

        if self.verbose:
            print('benchmark: ', self.benchmark)
            print('split type: ', self.splitType)
            print('base path: ', self.basePath)
            print('image path', self.imgPath)
            print('annotation path', self.annoPath)

        # initialize
        self.data = {'srcName':[],'trgName':[],'srcKps':[],'trgKps':[],'srcBbox':[],'trgBbox':[],'clsIdx':[],'flip':[]}

        self.loadData()
    

    def loadData(self):
        self.readPairFile()
        self.readAnnotations()
        #self.visualizer(5)
        
    
    def readPairFile(self):
        self.pairPath = os.path.join(self.basePath,self.splitType+'_pairs.csv')

        if self.verbose:
            print('begin loading ' + self.splitType + 'pair info..')
            print(self.pairPath)
        
        with open(self.pairPath,'r') as f:
            for i in f.readlines()[1:]:
                i = i.split(',')

                # extract pure imageFile name
                j = i[0].split('/')[-1]
                self.data['srcName'].append(j)
                j = i[1].split('/')[-1]
                self.data['trgName'].append(j)
                self.data['clsIdx'].append(int(i[2]))
                if self.splitType == 'train':
                    self.data['flip'].append(int(i[3]))
        
        if self.verbose:
            print('retrieved ' + str(len(self.data['srcName']))+' valid pairs')

    def readAnnotations(self):

        for i in range(len(self.data['srcName'])):
            srcKps,srcBbox = self.getMatValues(i,'src')
            
            #srcKps format [(x1,y1),(x2,y2),...] 
            trgKps,trgBbox = self.getMatValues(i,'trg')

            # mutual check for mismatch keypoint pair due to occlusion
            srcK = []
            trgK = []
            for srcKK, trgKK in zip(srcKps, trgKps):
                if len(torch.isnan(srcKK).nonzero()) != 0 or \
                        len(torch.isnan(trgKK).nonzero()) != 0:
                    continue
                else:
                    srcK.append(srcKK)
                    trgK.append(trgKK)
            
            self.data['srcKps'].append(torch.stack(srcK).t())
            #data['srckps'] format : [[x1,x2,..],[y1,y2,...]]
            self.data['trgKps'].append(torch.stack(trgK).t())
            self.data['srcBbox'].append(srcBbox)
            self.data['trgBbox'].append(trgBbox)
            
            if self.splitType == 'train' and self.data['flip'][i] == 1:
                a = self.loadImg(i,'src').shape[1]
                self.data['srcKps'][i][0] = a - self.data['srcKps'][i][0]
                self.data['srcBbox'][i][0] = a - self.data['srcBbox'][i][0]
                self.data['srcBbox'][i][2] = a - self.data['srcBbox'][i][2]
                b = self.loadImg(i,'trg').shape[1]
                self.data['trgKps'][i][0] = a - self.data['trgKps'][i][0]
                self.data['trgBbox'][i][0] = a - self.data['trgBbox'][i][0]
                self.data['trgBbox'][i][2] = a - self.data['trgBbox'][i][2]

    def getMatValues(self,index,dtype):

        imgName = self.data[dtype+'Name'][index]
        annoPath = os.path.join(self.annoPath,self.classNameList[self.data['clsIdx'][index]],imgName[:-4]+'.mat')
        Kps = self.loadMat(annoPath,'kps')
        Bbox = self.loadMat(annoPath,'bbox')
        Kps = torch.tensor(Kps).float()
        Bbox = torch.tensor(Bbox[0].astype(float))

        return Kps,Bbox

            

    def visualizer(self,index):


        # not available if two images dont correspond in dimension2
        typeName = 'src'
        srcImg = self.loadImg(index,typeName)
        srcKps = self.data[typeName+'Kps'][index]
        srcBbox = self.data[typeName+'Bbox'][index]

        typeName = 'trg'
        trgImg = self.loadImg(index,typeName)
        trgKps = self.data[typeName+'Kps'][index]
        trgBbox = self.data[typeName+'Bbox'][index]

        if self.verbose:
            print(srcImg.shape)
            print(trgImg.shape)

        jointImg = torch.cat((srcImg,trgImg),0)
        
        
        if self.verbose:
            print(jointImg.shape)
            print(srcKps.shape)
            print(self.data['flip'][index])
        
        for i in range(srcKps.shape[1]):
            xa = float(srcKps[0,i])
            ya = float(srcKps[1,i])
            xb = float(trgKps[0,i])
            yb = float(trgKps[1,i]) +srcImg.shape[0]




            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa,ya), radius=5, color=c))
            plt.gca().add_artist(plt.Circle((xb,yb), radius=5, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.5) 

        bboxa = self.data['srcBbox'][index]
        bboxb = self.data['trgBbox'][index]
        
        plt.gca().add_patch(plt.Rectangle((bboxa[0], bboxa[1]),bboxa[2] - bboxa[0] + 1,bboxa[3] - bboxa[1] + 1, fill=False,edgecolor='red', linewidth=2.5))
        plt.gca().add_patch(plt.Rectangle((bboxb[0], bboxb[1]+srcImg.shape[0]),bboxb[2] - bboxb[0] + 1,bboxb[3] - bboxb[1] + 1, fill=False,edgecolor='red', linewidth=2.5))

        plt.imshow(jointImg)
        plt.axis('off')
        # remove blank space around the pic
        plt.savefig(self.tmpPath+'/debug.jpg',bbox_inches = 'tight',pad_inches = 0.0)








    def loadMat(self, filePath, objName):

        matContents = sio.loadmat(filePath)
        matObj = matContents[objName]

        return matObj



    def loadImg(self, index, typeName = 'src'):

        imgType = typeName + 'Name'
        
        imgName = self.data[imgType][index]
        imgPath = os.path.join(self.imgPath,imgName)

        if self.verbose:
            print('trying to fetch image ', imgPath)

        img = mpimg.imread(imgPath)

        
        if self.splitType == 'train' and self.data['flip'][index] == 1:
            img = np.flip(img,1)
        
        img = torch.tensor(img.astype(np.float32)/256.)

        #print(img.shape)

        '''
        # horizontal flip
        for i in range(int(img.shape[1]/2)):
            tmp = img[:,img.shape[1]-i-1,:].clone()
            img[:,img.shape[1]-i-1,:] = img[:,i,:].clone()
            img[:,i,:] = tmp.clone()
        
        
        
        
        #display such image
        plt.imshow(img)
        plt.axis('off')
        # remove blank space around the pic
        plt.savefig(self.tmpPath+'/'+typeName+'debug.jpg',bbox_inches = 'tight',pad_inches = 0.0)
        '''

        return img




    def __getitem__(self, index):

        
        sample = dict()
        sample['srcName'] = self.data['srcName'][index]
        sample['trgName'] = self.data['trgName'][index]
        sample['pairClassIdx'] = self.data['clsIdx'][index]
        sample['pairClass'] = self.classNameList[self.data['clsIdx'][index]]
        sample['srcImg'] = self.loadImg(index,'src')
        sample['trgImg'] = self.loadImg(index,'trg')

        sample['srcKps'] = self.data['srcKps'][index]
        sample['trgKps'] = self.data['trgKps'][index]
        sample['srcBbox'] = self.data['srcBbox'][index]
        sample['trgBbox'] = self.data['trgBbox'][index]

        sample['dataLen'] = len(self.data['srcName'])

        if self.transform:
            sample = self.transform(sample)
        
        sample['srcImg'] = sample['srcImg'].to(self.device)
        sample['trgImg'] = sample['trgImg'].to(self.device)

        a = 0
        print(self.pckType)
        if self.pckType == 'bbox':
            v = sample['trgBbox']
            a = torch.max(v[3]-v[1],v[2]-v[0])
        elif self.pckType == 'img':
            v = sample['trgImg']
            a = torch.max(v[0],v[1])
        else:
            print('illegal pck criterior!')
            exit()
        
        sample['pckBound'] = a
        return sample

    def __len__(self):
       
        return len(self.data['srcName'])



if __name__ == "__main__":
    print("current executing in 'customizedDataset.py file'")
    cd = customizedDataset()
    s = cd.__getitem__(1)