import torch

import torch.nn as nn
import torch.nn.functional as F

import resNet_src as resnet

from functools import reduce
from operator import add

import matplotlib.pyplot as plt

class FeatureExtractor(nn.Module):

    def __init__(self, backbone = 'resnet101', device = "cuda:0",verbose = True, topK = 2):

        super(FeatureExtractor,self).__init__()

        self.device = device
        self.verbose = verbose

        self.backbone = None
        self.topK = topK

        if backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained = True).to(self.device)
            nbottlenecks = [3,4,23,3]
        else:
            print('unavailable backbone name')
            exit()
        
        self.bottleNeckIdx = reduce(add,list(map(lambda x: list(range(x)),nbottlenecks)))
        self.layerIdx = reduce(add, [[i+1] * x for i,x in enumerate(nbottlenecks)])

        self.backbone.eval()

    
    def forward(self,image):
        
        feats = []

        #layer 0
        feat = self.backbone.conv1.forward(image)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        #layer 1-4
        for hid, (bid,lid) in enumerate(zip(self.bottleNeckIdx, self.layerIdx)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)
        
            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
            
            feat += res
            

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        
        # output before gap operation

        featMap = feat

        x = self.backbone.avgpool(feat)
        x = torch.flatten(x,1)
        fc = self.backbone.fc(x)

        # get CAM
        logits = F.softmax(fc, dim = 1)
        scores, predLabels = torch.topk(logits, k = self.topK, dim = 1)

        predLabel = predLabels[0]

        bz, nc, h, w = featMap.size()

        outputCAM = []

        #multi-scale CAM needed ?

        for label in predLabel:
            
            cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(featMap.view(nc,h*w))
            cam = cam.view(1,1,h,w)


            #interpolate to higher resolution needed? Y 



            #normalization
            cam = (cam - cam.min()) / cam.max()
            outputCAM.append(cam)
        
        outputCAM = torch.stack(outputCAM,dim = 0)
        outputCAM = outputCAM.max(dim = 0)[0]

        

        return featMap, outputCAM




class CorrelationCalculator(nn.Module):
    def __init__(self, normalizeFlag = True):
        super(CorrelationCalculator,self).__init__()
        self.normFlag = normalizeFlag
        self.ReLU = nn.ReLU()

    def forward(self,featureA,featureB):
        b,c,hA,wA = featureA.size()
        b,c,hB,wB = featureB.size()

        featureA = featureA.view(b,c,hA*wA).transpose(1,2)
        featureB = featureB.view(b,c,hB*wB)

        correlationTensor = torch.bmm(featureA,featureB).view(b,hA,wA,hB,wB).unsqueeze(1)

        if self.normFlag:
            correlationTensor = self.L2Norm(self.ReLU(correlationTensor))

        return correlationTensor





    def L2Norm(self,feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)


class MyNet(nn.Module):
    def __init__(self,featureCnn = 'resnet101', device = "cuda:0",verbose = True,visualizerPath = ''):
        super(MyNet,self).__init__()

        self.featureExtractor = FeatureExtractor(backbone = featureCnn)
        self.correlationCalculator = CorrelationCalculator()

        self.featureExtractor.eval()
        self.device = device

        self.verbose = verbose

        self.visualizerPath = visualizerPath


    def forward(self, imgBatch):

        if self.verbose:
            print('original src image shape: ', imgBatch['srcImg'].shape)
            print('original trg image shape: ', imgBatch['trgImg'].shape)

        featureA, camA = self.featureExtractor(imgBatch['srcImg'])
        featureB, camB = self.featureExtractor(imgBatch['trgImg'])

        if self.verbose:
            print('src feature shape:', featureA.shape)
            print('src CAM shape:', camA.shape)
            print('trg feature shape:', featureB.shape)
            print('trg CAM shape:', camB.shape)


        correlationTensor = self.correlationCalculator(featureA,featureB)
        
        if self.verbose:
            print('correlation shape:', correlationTensor.shape)


        plt.imshow(camA[0].cpu().detach().numpy().squeeze())
        plt.axis('off')
        # remove blank space around the pic
        plt.savefig('cam+debug.jpg',bbox_inches = 'tight',pad_inches = 0.0)
        return True


if __name__ == "__main__":
    print('currently executing file model.py ...')