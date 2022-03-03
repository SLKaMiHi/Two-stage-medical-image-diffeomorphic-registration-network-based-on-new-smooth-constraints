import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from scipy import ndimage

# def JacboianDet(y_pred, sample_grid):
#
#
#     y_pred = F.pad(y_pred,pad=(1,0,1,0))
#     J = y_pred + sample_grid
#     dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
#     dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]
#
#     dfdx = dx
#     dfdy = dy
#
#     Jdet = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
#
#     return Jdet

def JacboianDet(y_pred, sample_grid):
    y_pred = F.pad(y_pred, pad=(1, 0, 1, 0, 1, 0))

    y_pred = y_pred.permute(0, 2, 3, 4, 1)
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet

def neg_Jdet_loss(y, grid):
    neg_Jdet = -1.0 * JacboianDet(y,grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear')

        return flow
class Resize_Nd_Volume():
    def __init__(self,volume,out_shape,order=3):

        self.volume=volume
        self.out_shape=out_shape
        self.order=order
    def resize_nd_volume(self):
        shape0=self.volume.shape
        assert (len(shape0)==len(self.out_shape))
        scale=[(self.out_shape[i]+0.)/shape0[i] for i in range(len(shape0))]
        out_volume=ndimage.interpolation.zoom(self.volume,scale,order=self.order)
        return out_volume

class SpatialTransformNearest(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                    size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                    size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                    size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest')

        return flow


class new_Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, flow):
        model = torch.nn.functional.avg_pool3d(flow, 3, 1)
        model = torch.nn.functional.pad(model, pad=(1, 1, 1, 1, 1, 1), mode="constant", value=0)
        return torch.mean((flow - model) ** 2)


