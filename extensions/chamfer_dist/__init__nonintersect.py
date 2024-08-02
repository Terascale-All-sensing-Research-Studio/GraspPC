# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch


import chamfer

global minindices_notiqg

# def get_min_indices():
#     return runner.newminindices

class ChamferFunctionUni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        device = torch.device("cuda:0")
        dist1, idx1 = chamfer.uni_forward(xyz1, xyz2)
        # print("saving idx1: ", idx1)
        ctx.save_for_backward(xyz1, xyz2, idx1)
        
        
        return dist1

    @staticmethod
    def backward(ctx, grad_dist1):
        #now has to take in 4 xyz inputs for 
        
        global minindices_notiqg
        xyz1, xyz2, dist_idx1 = ctx.saved_tensors
        # print("loading idx1: ", dist_idx1)
        
        grad_xyz1, grad_xyz2 = chamfer.uni_backward(xyz1, xyz2, dist_idx1, grad_dist1)
        
        grad_xyz1[minindices_notiqg] = 0
        grad_xyz2[minindices_notiqg] = 0
        # print("got here as well...")
        return grad_xyz1, grad_xyz2

class NonIntersectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, nxyz2):
        device = torch.device("cuda:0")
        dps1, dists1, idx1 = chamfer.niforward(xyz1, xyz2, nxyz2)
        # print("saving idx1: ", idx1)
        ctx.save_for_backward(xyz1, xyz2, nxyz2, dps1, idx1)
        
        return dps1

    @staticmethod
    def backward(ctx, grad_dps1):
        #now has to take in 4 xyz inputs for 
        
        global minindices_notiqg
        
        xyz1, xyz2, nxyz2, dps1, idx1 = ctx.saved_tensors
        # print("loading idx1: ", dist_idx1)
        
        grad_xyz1, grad_xyz2, grad_nxyz2 = chamfer.nibackward(xyz1, xyz2, nxyz2, idx1, grad_dps1)
        
        grad_xyz1[minindices_notiqg] = 0
        grad_xyz2[minindices_notiqg] = 0
        grad_nxyz2[minindices_notiqg] = 0
        # print("got here as well...")
        return grad_xyz1, grad_xyz2, grad_nxyz2

class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        device = torch.device("cuda:0")
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        # print("saving idx1: ", idx1)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        
        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        #now has to take in 4 xyz inputs for 
        
        global minindices_notiqg
        
        xyz1, xyz2, dist_idx1, dist_idx2 = ctx.saved_tensors
        # print("loading idx1: ", dist_idx1)
        
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, dist_idx1, dist_idx2, grad_dist1, grad_dist2)
        
        grad_xyz1[minindices_notiqg] = 0
        grad_xyz2[minindices_notiqg] = 0
        # print("got here as well...")
        return grad_xyz1, grad_xyz2

class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, _ = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) 

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
       
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.min_loss_index = None
    
    def set_min_loss(self, min_loss_indices):
        self.min_loss_index = min_loss_indices

    def forward(self, xyz1, xyz2):
       
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
          
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        #4 different dist 1 and 4 different dist2 
        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
  
        dist1 = torch.sqrt(dist1)

        dist2 = torch.sqrt(dist2)

        return (torch.mean(dist1,1) + torch.mean(dist2,1))/2
        
class ChamferDistanceL1Uni(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
       
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.min_loss_index = None
    
    def set_min_loss(self, min_loss_indices):
        self.min_loss_index = min_loss_indices

    def forward(self, xyz1, xyz2):
       
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
          
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        #4 different dist 1 and 4 different dist2 
        # dist1 = ChamferFunctionUni.apply(xyz1, xyz2)  
        dist1, _ = ChamferFunction.apply(xyz1,xyz2)  # Natasha changed this (11/13, 7am)

        dist1 = torch.sqrt(dist1)
        
        # following evenly weights all points, gives regular mean
        # weights = torch.ones( dist1.shape,dtype=torch.float32,device = dist1.get_device() ) 
        
        # following weights points by their distance, gives weighted mean
        # @Ava: need to bring in a value for dist1 and distthresh -- these are hyperparameters
        #    distthresh is the minimum distance as percentage of the bounding box
        #           below which we want points to be weighted
        #           more in the computation -- i.e., closer to the object surface
        #           something like 0.1 may work
        #    distmultiplier sharpens the activation to be similar to a step function, 
        #           set around 10-20 for reasonable behavior
        weights = torch.sigmoid( 10 * (dist1 - .01) )

        # @Ava sigmoid may not work, since it could cause the gradients to go to zero
        #   even for points that are near the object. In that case, try the ReLU
        #   function as below
        # weights = torch.nn.ReLU( (dist1 - 0.03) )
        # mod = torch.nn.ReLU()
        # weights = mod(dist1 - 0.03) 
        
        count = torch.sum(weights,1)
        count = count.unsqueeze(1)
        weights = torch.div(weights,count)

        gamma = .01

        return gamma * torch.sum( torch.mul( dist1, weights ), 1 )  # if weights is all 1s then same as torch.mean(dist1,1)      
        # return torch.mean(dist1,1)  # Natasha uncommented this (11/13, 7am)


class NonIntersect(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
       
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.min_loss_index = None
    
    def set_min_loss(self, min_loss_indices):
        self.min_loss_index = min_loss_indices

    def forward(self, xyz1, xyz2, nxyz2):
       
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
          
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dps1 = NonIntersectFunction.apply(xyz1,xyz2,nxyz2)
       
        dps1t = torch.zeros(dps1.shape, dtype=torch.float32, device=dps1.get_device())

        idxgood = dps1 > 0
        dps1t[idxgood] = dps1[idxgood]    

        w = 5

        dps1e = torch.exp( dps1t  * w )

        # weights = torch.sigmoid( 10 * (dist1 - .01) )

        # count = torch.sum(weights,1)
        # count = count.unsqueeze(1)
        # weights = torch.div(weights,count)

        gamma = .02
        return gamma * torch.mean( dps1e,1 )

        # return gamma * torch.sum( torch.mul( dist1, weights ), 1 )  # if weights is all 1s then same as torch.mean(dist1,1)      
        # return torch.mean(dist1,1)  # Natasha uncommented this (11/13, 7am)


class ChamferDistanceL1Coarse(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
       
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.min_loss_index = None
    
    def set_min_loss(self, min_loss_indices):
        self.min_loss_index = min_loss_indices

    def forward(self, xyz1, xyz2):
       
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
          
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        #4 different dist 1 and 4 different dist2 
        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        #coarse 
        alpha =  1  # @Ava: run wih alpha = 1 on tars4 and alpha = 0.25 on tars 7
        beta = 1
        dist1 = torch.sqrt(dist1)

        dist2 = torch.sqrt(dist2)

        return (alpha * (torch.mean(dist1,1)) + beta * (torch.mean(dist2,1)))/2

class ChamferDistanceL1Dense(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
       
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.min_loss_index = None
    
    def set_min_loss(self, min_loss_indices):
        self.min_loss_index = min_loss_indices

    def forward(self, xyz1, xyz2):
       
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
          
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        #4 different dist 1 and 4 different dist2 
        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        
        dist1 = torch.sqrt(dist1)

        dist2 = torch.sqrt(dist2)

        return (torch.mean(dist1,1) +  torch.mean(dist2,1))/2





class ChamferDistanceL1_PM(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, _ = ChamferFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        return torch.mean(dist1)



class MinIndicesManager:
    def __init__(self):
        self.minindices = None

    def set_minindices(self, minminindices):
        global minindices_notiqg
        minindices_notiqg = minminindices
    
        
        