import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import resNet_src as resnet

from functools import reduce
from operator import add

import matplotlib.pyplot as plt

from utils_sc import *

class FeatureExtractor(nn.Module):

    def __init__(self, backbone = 'resnet101', device = "cuda:0",verbose = True, topK = 2):

        super(FeatureExtractor,self).__init__()

        self.device = device
        self.verbose = verbose

        self.backbone = None
        self.topK = topK

        # SCOT layers for ResNet 101
        self.hyperpixelIds= [2,22,24,25,27,28,29]

        # Hyperpixel Flow
        #self.hyperpixelIds = [2,17,21,22,25,26,28]

        self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16]).to(device)
        self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)


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
        
        if 0 in self.hyperpixelIds:
            feats.append(feat.clone())

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

            if hid + 1 in self.hyperpixelIds:
                feats.append(feat.clone())

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat_tmp in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat_tmp, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)


        
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

            #print('feats shape: ',feats.shape)
            #interpolate to higher resolution needed? Y 
            cam = F.interpolate(cam, (feats.shape[2],feats.shape[3]), None, 'bilinear', True)[0,0]


            #normalization
            cam = (cam - cam.min()) / cam.max()
            outputCAM.append(cam)
        
        outputCAM = torch.stack(outputCAM,dim = 0)
        outputCAM = outputCAM.max(dim = 0)[0]

        rfsz = self.rfsz[self.hyperpixelIds[0]]
        jsz = self.jsz[self.hyperpixelIds[0]]

        hpgeometry = receptive_fields(rfsz, jsz, feats.size()).to(self.device)

        return hpgeometry,feats, outputCAM




class CorrelationCalculator(nn.Module):
    def __init__(self, normalizeFlag = True,verbose = True):
        super(CorrelationCalculator,self).__init__()
        self.normFlag = normalizeFlag
        self.ReLU = nn.ReLU()
        self.verbose = verbose

    def forward(self,featureA,featureB):
        b,c,hA,wA = featureA.size()
        b,c,hB,wB = featureB.size()

        featureAnorms = torch.norm(featureA, p = 2, dim = 1).transpose(1,2)
        featureBnorms = torch.norm(featureB, p = 2, dim = 1)

        featureA = featureA.view(b,c,hA*wA)
        featureB = featureB.view(b,c,hB*wB)
        
        featureAnorms = torch.norm(featureA, p = 2, dim = 1).transpose(0,1)
        featureBnorms = torch.norm(featureB, p = 2, dim = 1)

        featureA = featureA.transpose(1,2)
        
        correlationTensor = torch.bmm(featureA,featureB)/torch.matmul(featureAnorms, featureBnorms)
        


        if self.verbose:
            print('featureA shape: ',featureA.shape)
            print('featureB shape: ',featureB.shape)
            print('norm featureA shape: ', featureAnorms.shape)
            print('norm featureB shape: ', featureBnorms.shape)
            print('correlation tensor shape: ',correlationTensor.shape)
            print('multiplication result shape : ', torch.matmul(featureAnorms, featureBnorms).shape)
            #print('correlation tensor: ', correlationTensor)

        


        if self.normFlag:
            correlationTensor = self.ReLU(correlationTensor)
        

        return correlationTensor,(hB, wB)





    def L2Norm(self,feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

    
    def inferFromCorrelation(self,correlationMap,srcImgShape,srcFeatureShape,trgFeatureShape):

        if self.verbose:
            print('correlation map shape: ', correlationMap.shape)
            print('feature shape:', srcFeatureShape)
            print('image shape:', srcImgShape)
            #print(correlationMap)

        shape = correlationMap.shape
        #correlationMap = correlationMap.view(shape[0],shape[1],shape[2]*shape[3],shape[4]*shape[5])
        
        if self.verbose:
            print('correlation map shape:', correlationMap.shape)

        inference = correlationMap.argmax(dim = 1).squeeze(0)

        print('correlationMap shape: ', correlationMap.shape)
        print('inference shape: ', inference.shape)
        #exit()
        inferTmp = []
        
        for k,i in enumerate(inference):
            h = i / srcFeatureShape[2] *(float(srcImgShape[1])/srcFeatureShape[2]) + (float(srcImgShape[1])/srcFeatureShape[2]) * 0.5
            x = int(i / srcFeatureShape[3])
            w = float(i - x * srcFeatureShape[2]) * float(srcImgShape[2])/srcFeatureShape[3] + float(srcImgShape[2])/srcFeatureShape[3] * 0.5
            #x = int(k / trgFeatureShape[3])
            #print(k)
            #print(x)
            
            #inferTmp.append([(k - x * trgFeatureShape[3])* float(srcImgShape[2])/trgFeatureShape[3],x * (float(srcImgShape[1])/trgFeatureShape[2])])
            inferTmp.append([(i - x * srcFeatureShape[3])* float(srcImgShape[2])/srcFeatureShape[3],x * (float(srcImgShape[1])/srcFeatureShape[2])])
        #exit()
        print(trgFeatureShape)
        #exit()
        inference = torch.from_numpy(np.array(inferTmp,dtype = 'float64')).view(trgFeatureShape[2],trgFeatureShape[3],-1)


        if self.verbose:
            print('inference shape: ',inference.shape)
            #print(inference)
        
        return inference

class transportSolver(nn.Module):

    def __init__(self, iterTimes = 10, verbose = True, epsilon = 0.05):
        
        super(transportSolver,self).__init__()
        self.iterTimes = iterTimes
        self.verbose = verbose
        self.epsilon = epsilon

        if self.verbose:
            print('iteration times: ',self.iterTimes)
    
    def forward(self,costMatrix,marginals):

        PI, _, _, _ = self.sinkhornSolver(costMatrix,marginals) 
        if self.verbose:
            print('PI shape: ',PI.shape)
        return PI
    
    def sinkhornSolver(self, costMatrix, marginals, warm = False, tol = 10e-9):
        # SCOT rhm_map.py

        C = costMatrix.squeeze(0)

        nu = marginals[1].unsqueeze(1)
        mu = marginals[0].unsqueeze(1)

        niter = self.iterTimes

        if self.verbose:
            print('C shape: ',C.shape)
            print('mu shape: ',mu.shape)
            print('nu shape: ',nu.shape)
            print('torch.mm(K,b) shape: ',torch.mm(C, nu).shape)

        a = []

        if not warm:
            a = torch.ones((C.shape[0],1)) / C.shape[0]
            a = a.cuda()


        K = torch.exp(-C/self.epsilon)

        Err = torch.zeros((niter,2)).cuda()
        
        for i in range(niter):
            b = nu/torch.mm(K.t(), a)
            if i%2==0:
                Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
                if i>0 and (Err[i,0]) < tol:
                    break

            a = mu / torch.mm(K, b)

            if i%2==0:
                Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
                if i>0 and (Err[i,1]) < tol:
                    break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

        del a; del b; del K
        return PI,mu,nu,Err
        

class MyNet(nn.Module):
    def __init__(self,featureCnn = 'resnet101', device = "cuda:0",verbose = True,visualizerPath = ''):
        super(MyNet,self).__init__()


        self.verbose = verbose
        self.featureExtractor = FeatureExtractor(backbone = featureCnn,verbose = self.verbose)
        self.correlationCalculator = CorrelationCalculator(verbose = self.verbose)
        self.transportSolver = transportSolver(verbose = self.verbose)

        self.featureExtractor.eval()
        self.device = device

        self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16]).to(device)
        self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)

        self.visualizerPath = visualizerPath


    def forward(self, imgBatch):

        if self.verbose:
            print('original src image shape: ', imgBatch['srcImg'].shape)
            print('original trg image shape: ', imgBatch['trgImg'].shape)

        src_geometry, featureA, camA = self.featureExtractor(imgBatch['srcImg'])
        trg_geometry,featureB, camB = self.featureExtractor(imgBatch['trgImg'])

        if self.verbose:
            print('src feature shape:', featureA.shape)
            print('src CAM shape:', camA.shape)
            print('trg feature shape:', featureB.shape)
            print('trg CAM shape:', camB.shape)


        correlationTensor, trgImgShape = self.correlationCalculator(featureA,featureB)

        PI = self.transportSolver(1-correlationTensor,[camA.view(-1),camB.view(-1)]).unsqueeze(0)
        
        #print('correlation tensor shape: ',correlationTensor.shape)

        correlationTensor = PI

        inference = self.correlationCalculator.inferFromCorrelation(correlationTensor,srcFeatureShape = featureA.shape,srcImgShape = imgBatch['srcImg'][0].shape,trgFeatureShape = featureB.shape)

        # transport problem result
        

        #print('inference shape', inference.shape)
        
        #warp_image(imgBatch['srcImg'][0,:,:,:], inference,trgImgShape = imgBatch['trgImg'][0].shape, filePath = self.visualizerPath,verbose = self.verbose)

        
        if self.verbose:
            print('correlation shape:', correlationTensor.shape)


        
        if self.visualizerPath != '':
            plt.imshow(camA.cpu().detach().numpy().squeeze())
            plt.axis('off')
            # remove blank space around the pic
            plt.savefig(self.visualizerPath+'/cam+src+debug.jpg',bbox_inches = 'tight',pad_inches = 0.0)
            self.camImage(camA,imgBatch['srcImg'][0,:,:,:],'src')
            self.camImage(camB,imgBatch['trgImg'][0,:,:,:],'trg')
        
        return inference, src_geometry, trg_geometry

    def camImage(self,cam,origImg,note = 'src'):
        origImg = origImg.permute(1,2,0)
        cam = cam.unsqueeze(0).unsqueeze(0)
        #print(cam.shape)
        sz = origImg.size()
        if self.verbose:
            print('origImg size:', sz)
            print('cam size:', cam.size())
        camUpSampled = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)
        if self.verbose:
            print('cam upsampled size:', camUpSampled.size())
        plt.cla()
        plt.imshow(camUpSampled.cpu().detach().numpy().squeeze(), alpha = 0.9)
        #plt.imshow(origImg.cpu().detach().numpy().squeeze())
        plt.axis('off')
        plt.savefig(self.visualizerPath+'/'+note+'cam_upSample.jpg',bbox_inches = 'tight',pad_inches = 0.0)
        plt.imshow(origImg.cpu().detach().numpy().squeeze(),alpha = 0.2)
        plt.savefig(self.visualizerPath+'/'+note+'camUpwithOrigImg.jpg',bbox_inches = 'tight',pad_inches = 0.0)







if __name__ == "__main__":
    print('currently executing file model.py ...')
    mn = MyNet(visualizerPath = 'visualizer')
