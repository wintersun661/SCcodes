import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt

def normalize_axis(x,L): 
    # x (1,L) -> (-1,1)
    return 2*(x-1- (L-1)/2.)/(L-1)

def unnormalize_axis(x,L):
    return x*(L-1)/2+1+(L-1)/2

def neighbours(box, kps):
    r"""Returns boxes in one-hot format that covers given keypoints"""
    box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1).to('cuda:0')
    kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1).to('cuda:0')

    xmin = kps_duplicate[0].ge(box_duplicate[0])
    ymin = kps_duplicate[1].ge(box_duplicate[1])
    xmax = kps_duplicate[0].le(box_duplicate[2])
    ymax = kps_duplicate[1].le(box_duplicate[3])

    nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()
    n_neighbours = nbr_onehot.sum(dim=1)

    return nbr_onehot, n_neighbours

def receptive_fields(rfsz, jsz, feat_size):

    r"""Returns a set of receptive fields (N, 4)"""

    # Hyper=pixel flow
    width = feat_size[2]
    height = feat_size[1]

    feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2)
    feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

    box = torch.zeros(feat_ids.size()[0], 4)
    box[:, 0] = feat_ids[:, 1] * jsz - rfsz // 2
    box[:, 1] = feat_ids[:, 0] * jsz - rfsz // 2
    box[:, 2] = feat_ids[:, 1] * jsz + rfsz // 2
    box[:, 3] = feat_ids[:, 0] * jsz + rfsz // 2

    return box


def predict_kps(src_box, trg_box, src_kps, confidence_ts):
    r"""Transfer keypoints by nearest-neighbour assignment"""

    # 1. Prepare geometries & argmax target indices
    print('confidence shape: ',confidence_ts.shape)
    _, trg_argmax_idx = torch.max(confidence_ts, dim=1)
    print('trf_argmax_idx shape: ',trg_argmax_idx.shape)
    print('src_box shape: ',src_box.shape)
    print('src kps shape: ',src_kps.shape)
    src_geomet = src_box[:, :2].unsqueeze(0).repeat(len(src_kps.t()), 1, 1)
    trg_geomet = trg_box[:, :2].unsqueeze(0).repeat(len(src_kps.t()), 1, 1)

    # 2. Retrieve neighbouring source boxes that cover source key-points
    src_nbr_onehot, n_neighbours = neighbours(src_box, src_kps)
    print('src_nbr_onehot shape: ', src_nbr_onehot.shape)

    # 3. Get displacements from source neighbouring box centers to each key-point
    src_displacements = src_kps.t().unsqueeze(1).repeat(1, len(src_box), 1).to('cuda:0') - src_geomet.to('cuda:0')
    src_displacements = src_displacements * src_nbr_onehot.unsqueeze(2).repeat(1, 1, 2).float()

    # 4. Transfer the neighbours based on given confidence tensor
    vector_summator = torch.zeros_like(src_geomet)
    print('src geometry shape: ', src_geomet.shape)
    
    src_idx = src_nbr_onehot.nonzero()   
    print('src_idx shape: ',src_idx.shape)         
    print('trg_geomet shape: ',trg_geomet.shape)                                         
    trg_idx = trg_argmax_idx.index_select(dim=1, index=src_idx[:, 1])
    print('trg_idx shape: ', trg_idx.shape)
    vector_summator[src_idx[:, 0], src_idx[:, 1]] = trg_geomet[src_idx[:, 0], trg_idx]
    vector_summator += src_displacements
    pred = (vector_summator.sum(dim=1) / n_neighbours.unsqueeze(1).repeat(1, 2).float())

    return pred.t()

def cal_pck(kps_gr,kps_pred,reference,alpha = 0.1,verbose = True):
    #print(kps_gr.shape)
    l2dist = torch.pow(torch.sum(torch.pow(kps_gr - kps_pred, 2).squeeze(0), 0), 0.5)
    
    thres = reference.expand_as(l2dist).float()
    correct_pts = torch.le(l2dist, reference * alpha)
    if verbose:
        print(kps_gr.shape)
        print(l2dist.shape)
        print(reference*alpha)
        print(l2dist)
    return (torch.sum(correct_pts)/float(kps_gr.shape[2])* 100).cpu().numpy()

def warp_image(image, flow, trgImgShape,filePath,verbose = True):
    """
    Warp image (np.ndarray, shape=[h_src,w_src,3]) with flow (np.ndarray, shape=[h_tgt,w_tgt,2])
    # DCCNet
    
    """
    if verbose:
        print('preparing for warpping image...')
        print('flow shape: ', flow.shape)
        #print(flow)
        print('src image shape: ',image.shape)
    h_src,w_src=image.shape[1],image.shape[2]
    #print('src image shape:', image.shape)
    flow = F.interpolate(flow.permute(2,0,1).unsqueeze(0), (trgImgShape[1],trgImgShape[2]), None, 'bilinear', True)
    #print(flow.permute(0,2,3,1))
    #print('flow shape,', flow.shape)
    flow = flow.squeeze(0)
    
    if verbose:
        print('bilinear interpolate flow shape: ',flow.shape)
    sampling_grid_torch = np_flow_to_th_sampling_grid(flow.permute(1,2,0), h_src, w_src)
    if verbose:
        #print(sampling_grid_torch)
        print('sampling grid torch shape: ',sampling_grid_torch.shape)
    image_torch = image.unsqueeze(0)
    warped_image_torch = F.grid_sample(image_torch, sampling_grid_torch)
    
    #warped_image_torch = F.interpolate(warped_image_torch, (trgImgShape[1],trgImgShape[2]), None, 'bilinear', True)
    #print(warped_image_torch.permute(0,2,3,1))
    if verbose:
        print('warped image shape: ',warped_image_torch.shape)
    plt.cla()
    plt.imshow(warped_image_torch.squeeze(0).permute(1,2,0).cpu().detach().numpy())
    plt.axis('off')
    # remove blank space around the pic
    plt.savefig(filePath+'/warped_src.jpg',bbox_inches = 'tight',pad_inches = 0.0)
    return warped_image_torch

def warp_kps(tgrKps_gt,inference,trgImgShape,verbose = True):
    if verbose:
        print('inference shape: ', inference.shape)
    inference = F.interpolate(inference.permute(2,0,1).unsqueeze(0), (trgImgShape[1],trgImgShape[2]), None, 'bilinear', True).squeeze(0)
    if verbose:
        print('inference shape: ', inference.shape)
    srcKps_pred = []
    num = tgrKps_gt.shape[1]
    for i in range(num):
        
        y = tgrKps_gt[0,i]
        x = tgrKps_gt[1,i]
        x_down = int(x) 
        y_down = int(y) 
        
        x_up = x_down+1
        y_up = y_down+1

        flagX = False
        flagY = False

        if x_up >= inference.shape[1]:
            flagX = True
            
        if y_up >= inference.shape[2]:
            flagY = True
            
        if flagX and flagY:
            result = inference[:,x_down,y_down]
        elif flagX:
            result = linear_interpolation(y_down,y_up,inference[:,x_down,y_down],inference[:,x_down,y_up],y)
        elif flagY:
            result = linear_interpolation(x_down,x_up,inference[:,x_down,y_down],inference[:,x_up,y_down],x)
        else:
            # nearest point value  assignment on edge point
            p1 = (x_down,y_down,inference[:,x_down,y_down])
            p2 = (x_up,y_down,inference[:,x_up,y_down])
            p3 = (x_down,y_up,inference[:,x_down,y_up])
            p4 = (x_up,y_up,inference[:,x_up,y_up])
            points = [p1,p2,p3,p4]
        
            result = bilinear_interpolation(x,y,points)
            #print(result)
        srcKps_pred.append(result)
    r = torch.stack(srcKps_pred).t()
    #print(r.shape)
    return r

def linear_interpolation(x1: float, x2: float, y1: float, y2: float, x: float):
    """Perform linear interpolation for x between (x1,y1) and (x2,y2) """

    return ((y2 - y1) * x + x2 * y1 - x1 * y2) / (x2 - x1)

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    #print(points)
    points = sorted(points)               # order points by x, then by y 
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def np_flow_to_th_sampling_grid(flow,h_src,w_src,device = "cuda:0"):

    #DCCNet
    h_tgt,w_tgt=flow.shape[0],flow.shape[1]
    grid_x,grid_y = np.meshgrid(range(1,w_tgt+1),range(1,h_tgt+1))
    disp_x=flow[:,:,0]
    disp_y=flow[:,:,1]
    #source_x=grid_x+disp_x
    #source_y=grid_y+disp_y
    source_x = disp_x
    source_y = disp_y
    source_x_norm=normalize_axis(source_x,w_src) 
    source_y_norm=normalize_axis(source_y,h_src) 
    sampling_grid=np.concatenate((np.expand_dims(source_x_norm,2),
                                  np.expand_dims(source_y_norm,2)),2)
    sampling_grid_torch = Variable(torch.FloatTensor(sampling_grid).unsqueeze(0))
    sampling_grid_torch = sampling_grid_torch.to(device)
    return sampling_grid_torch