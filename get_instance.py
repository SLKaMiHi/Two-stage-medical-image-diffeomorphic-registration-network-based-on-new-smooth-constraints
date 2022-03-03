import warnings
import torch
import torch.nn as nn
import os
import time
import numpy as np
import losses
import glob
import pandas as pd
import pystrum.pynd.ndutils as nd
import torch
import torch.nn.functional as F
import pystrum.pynd.ndutils as nd
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from time import *
from model import  Net
from torchinfo import summary
from ptflops import get_model_complexity_info
import SimpleITK as sit
from module import SpatialTransformNearest,generate_grid


checkpoint_file = "/home/mamingrui/songlei/ablation/CNN_Jac/V1/new_grad/jac_half/checkpoint&log0.1*100/BN_False/lsMSElr5e-05/s0.1/norm_model_best.pth.tar"
vol_orig_shape = (96, 112, 96)
model = Net(3).cuda()

check_point = torch.load(checkpoint_file, map_location='cpu')
check_point['state_dict']
model.load_state_dict(check_point['state_dict'])

model.eval()

atlas="/home/mamingrui/data/None_atlas_reg/valsets_all/OASIS_OAS1_0456_MR1.npy"
atlas_label = "/home/mamingrui/data/None_atlas_reg/valsets_label_all/OASIS_OAS1_0456_MR1_label.npy"
val = "/home/mamingrui/data/None_atlas_reg/valsets_all/OASIS_OAS1_0434_MR1.npy"
val_label = "/home/mamingrui/data/None_atlas_reg/valsets_label_all/OASIS_OAS1_0434_MR1_label.npy"
STN = SpatialTransformNearest()
grid = generate_grid((97, 113, 97))
grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
fixed_img = torch.Tensor(np.load(atlas)).cuda()
fixed_img = fixed_img.unsqueeze(0).unsqueeze(0)
fixed_img_label = np.load(atlas_label)

moved_img = torch.Tensor(np.load(val)).cuda()
moved_img = moved_img.unsqueeze(0).unsqueeze(0)
moved_img_label = torch.Tensor(np.load(val_label)).cuda()
moved_img_label = moved_img_label.unsqueeze(0).unsqueeze(0)

X_Y,F_X_Y, atlas_half, pred_half, flow_half = model(moved_img, fixed_img, grid)

warped_B = X_Y.detach().cpu().numpy()[0, 0, :, :, :]

warped_B_label = STN(moved_img_label,F_X_Y.permute(0, 2, 3, 4, 1), grid[:, 0:96, 0:112, 0:96, :]).data.cpu().numpy()[0, 0, :, :, :]
F_AB = F_X_Y.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, :, :, :, :]
sit.WriteImage(sit.GetImageFromArray(F_AB), "flow_My1.nii.gz")
sit.WriteImage(sit.GetImageFromArray(warped_B), "Moved_My1.nii.gz")
sit.WriteImage(sit.GetImageFromArray(warped_B_label), "Moved_label_My1.nii.gz")
# np.save("moved_diff.npy",warped_B)
# np.save("moved_label_diff.npy",warped_B_label)