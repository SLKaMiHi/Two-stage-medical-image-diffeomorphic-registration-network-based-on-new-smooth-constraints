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
from model import Net
from torchinfo import summary
from ptflops import get_model_complexity_info
from module import SpatialTransformNearest
from torch.autograd import Variable

def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    :return a list as the label length
    """

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid


class Fine_Tune_model(nn.Module):
    def __init__(self):
        super(Fine_Tune_model, self).__init__()
        self.input = nn.Sequential(nn.Conv3d(3, 16, 1, 1, bias=True), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv3d(16, 16, 5, 2, 2, bias=True), nn.ReLU())
        self.conv5_1 = nn.Sequential(nn.Conv3d(16, 32, 3, 1, 1, bias=True), nn.ReLU())
        self.conv5_2 = nn.Sequential(nn.ConvTranspose3d(32, 16, 2, 2, bias=True), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv3d(16, 16, 3, 2, 1, bias=True), nn.ReLU())
        self.conv3_1 = nn.Sequential(nn.Conv3d(16, 32, 3, 1, 1, bias=True), nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.ConvTranspose3d(32, 16, 2, 2, bias=True), nn.ReLU())

        self.stem_1 = nn.Conv3d(32, 16, 3, 1, 1, bias=True)
        self.delta = nn.Conv3d(16, 3, 3, 1, 1, bias=True)

        nd = Normal(0, 1e-5)
        self.delta.weight = nn.Parameter(Normal(0, 1e-5).sample(self.delta.weight.shape))
        self.delta.bias = nn.Parameter(torch.zeros(self.delta.bias.shape))


    def forward(self, x):
        x = self.input(x)
        x_1 = self.conv5(x)
        x_1 = self.conv5_1(x_1)
        x_1 = self.conv5_2(x_1)
        x_2 = self.conv3(x)
        x_2 = self.conv3_1(x_2)
        x_2 = self.conv3_2(x_2)
        x = torch.cat([x_1, x_2], 1)

        x = self.stem_1(x)

        delta = self.delta(x)

        return delta



csv_path = '/home/mamingrui/PycharmProjects/result_csv/songlei/0.4*500/'

labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
          20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]

label_name = ['Left-Cerebral-W. Matter',
              'Left-Cerebral-Cortex',
              'Left-Lateral-Ventricle',
              'Left-Inf-Lat-Ventricle',
              'Left-Cerebellum-W. Matter',
              'Left-Cerebellum-Cortex',
              'Left-Thalamus',
              'Left-Caudate',
              'Left-Putamen',
              'Left-Pallidum',
              '3rd-Ventricle',
              '4th-Ventricle',
              'Brain-Stem',
              'Left-Hippocampus',
              'Left-Amygdala',
              'Left-Accumbens',
              'Left-Ventral-DC',
              'Left-Vessel',
              'Left-Choroid-Plexus',
              'Right-Cerebral-W. Matter',
              'Right-Cerebral-Cortex',
              'Right-Lateral-Ventricle',
              'Right-Inf-Lat-Ventricle',
              'Right-Cerebellum-W. Matter',
              'Right-Cerebellum-Cortex',
              'Right-Thalamus',
              'Right-Caudate',
              'Right-Putamen',
              'Right-Pallidum',
              'Right-Hippocampus',
              'Right-Amygdala',
              'Right-Accumbens',
              'Right-Ventral-DC',
              'Right-Vessel',
              'Right-Choroid-Plexus']
vol_orig_shape = (97, 113, 97)

atlases = sorted(glob.glob("/home/mamingrui/songlei/OAS_new/atlases_new/*.npy"))
atlases_label = sorted(glob.glob("/home/mamingrui/songlei/OAS_new/atlases_label/*.npy"))
valsets = sorted(glob.glob("/home/mamingrui/songlei/OAS_new/valsets_new/*.npy"))
valsets_label = sorted(glob.glob("/home/mamingrui/songlei/OAS_new/valsets_label/*.npy"))



df_acc = pd.DataFrame(columns=['atlas_data', 'val_data', 'labels', 'label_name', 'dsc'])
df_jac = pd.DataFrame(columns=['atlas_data', 'val_data', 'jac_num'])
df_time = pd.DataFrame(columns=['atlas_data', 'val_data', 'execute_time'])
df_acc_subj = pd.DataFrame(columns=['atlas_data', 'val_data', 'dsc'])

checkpoint_file = "/home/mamingrui/songlei/ablation/CNN_Jac/V1/new_grad/jac_half/checkpoint&log0.4*500/BN_False/lsMSElr5e-05/s0.4/norm_model_best.pth.tar"

model = Net(3).cuda()

check_point = torch.load(checkpoint_file, map_location='cpu')
check_point['state_dict']
model.load_state_dict(check_point['state_dict'])

model.eval()
# summary(model,[(1,1,96,112,96),(1,1,96,112,96)])

from thop import profile,clever_format
from torchstat import stat





STN = SpatialTransformNearest()

jac_acc_sum = 0.0
val_acc_sum = 0.0
acc_total = []
vol_length = len(valsets)

"""AFT"""
AFT=False
model2 = Fine_Tune_model().cuda()
checkpoint_file='/home/mamingrui/PycharmProjects/formal_version/proposed_64_5_cycle/fine_tune3/CVPR/checkpoint_new_1105/fine_tune_72000.pth'

check_point = torch.load(checkpoint_file, map_location='cpu')
model2.load_state_dict(check_point)
grid = generate_grid((97, 113, 97))
grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
with torch.no_grad():
    for atlas, atlas_label in zip(atlases, atlases_label):
        atlas_name = os.path.splitext(os.path.split(atlas)[1])[0]

        # print(atlas, atlas_label)
        fixed_img = torch.Tensor(np.load(atlas)).cuda()
        fixed_img = fixed_img.unsqueeze(0).unsqueeze(0)
        fixed_img_label = np.load(atlas_label)

        acc_list = []
        jac_list = []
        temp = 0.0
        for val, val_label in zip(valsets, valsets_label):
            start_time = time()
            # print(val, val_label)
            moved_img = torch.Tensor(np.load(val)).cuda()
            moved_img = moved_img.unsqueeze(0).unsqueeze(0)
            moved_img_label = torch.Tensor(np.load(val_label)).cuda()
            moved_img_label = moved_img_label.unsqueeze(0).unsqueeze(0)
            # stat(model, (96,112,96))
            # flops, params = profile(model, inputs=(moved_img, fixed_img))
            # macs, params = clever_format([flops, params], "%.3f")
            # print(macs,params)
            # summary(model,moved_img,fixed_img)
            # macs, params = get_model_complexity_info(model, (1, 96,112,96), as_strings=True,
            #                                          print_per_layer_stat=True, verbose=True)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            X_Y, F_X_Y, atlas_half, pred_half, flow_half = model(moved_img, fixed_img,grid)

            """AFT"""
            # AFT
            if AFT == True:
                diff_AB = model2(F_X_Y)
                F_X_Y = F_X_Y - diff_AB


            # warped_A = transform(fixed_img, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0,
            #            0, :, :, :]
            warped_B = X_Y.detach().cpu().numpy()[0, 0, :, :, :]
            execute_time = (time() - start_time)

            F_AB = F_X_Y.permute(0, 2, 3, 4, 1).detach().cpu().numpy()[0, :, :, :, :]
            F_AB = F_AB.astype(np.float32)

            val_name = os.path.splitext(os.path.split(val)[1])[0]

            df_time.loc[len(df_time)] = [atlas_name, val_name, execute_time]

            # save_flow(F_BA, savepath + '/wrapped_flow_B_to_A.nii.gz')
            # save_flow(F_AB, savepath + '/wrapped_flow_A_to_B.nii.gz')
            # save_img(warped_B, savepath + '/wrapped_norm_B_to_A.nii.gz')
            # save_img(warped_A, savepath + '/wrapped_norm_A_to_B.nii.gz')

            # print("Finished.")
            """AFT"""
            ##origin registration
            # warped_B_label = STN(moved_img_label, F_X_Y,
            #                      grid,'nearest').data.cpu().numpy()[0, 0, :, :, :]
            # jac_det = jacobian_determinant((F_X_Y.permute(0, 2, 3, 4, 1) ).squeeze(0).detach().cpu())
            ##AFT registration
            warped_B_label = STN(moved_img_label,F_X_Y.permute(0, 2, 3, 4, 1), grid[:, 0:96, 0:112, 0:96, :]).data.cpu().numpy()[0, 0, :, :, :]
            # warped_B_label = moved_img_label.data.cpu().numpy()[0, 0, :, :, :]
            jac_det = jacobian_determinant((F_X_Y.permute(0, 2, 3, 4, 1)).squeeze(0).detach().cpu())

            jac_neg_item = np.sum(jac_det <= 0)
            # print(f'this validation pairs jac_neg_det is {jac_neg_item}')
            df_jac.loc[len(df_jac)] = [atlas_name, val_name, jac_neg_item]
            print(val_name)
            # warped_B_label=warped_B_label.squeeze(0).squeeze(0).detach().cpu().numpy()
            dice_score = dice(warped_B_label, fixed_img_label, labels)
            df_acc_subj.loc[len(df_acc_subj)] = [atlas_name, val_name, np.mean(dice_score)]
            acc_total.append(np.mean(dice_score))
            # dice_score_combine = []
            # for i in range(1, len(labels) + 1):
            #     if 11 <= i <= 13:
            #         dice_score_combine.append(dice_score[i - 1])
            #     elif i <= 10:
            #         dice_score_combine.append((dice_score[i - 1] + dice_score[i + 19 - 1]) / 2)
            #     elif 14 <= i <= 19:
            #         dice_score_combine.append((dice_score[i - 1] + dice_score[i + 16 - 1]) / 2)
            #     elif i >= 20:
            #         break
            # print(np.sum(dice_score) / len(labels))
            # print(len(dsc))
            # print(len(labels))
            # print(len(label_name))
            # print(atlas_name, val_name)

            for i, dsc in enumerate(dice_score):
                # print(i)
                df_acc.loc[len(df_acc)] = [atlas_name, val_name, labels[i], label_name[i], dsc]
                val_acc = np.sum(acc_list) / vol_length

df_acc.to_csv(csv_path + '3DVM_seg_atlas_validation_name_new.csv', index=False)
df_time.to_csv(csv_path + '3DVM_execute_time_new.csv', index=False)
df_jac.to_csv(csv_path + '3DVM_jac_new.csv', index=False)
df_acc_subj.to_csv(csv_path + '3DVM_acc_subj_new.csv', index=False)

df_time.to_csv(csv_path + '3DVM_execute_time_final.csv', index=False)
"""AFT"""
if AFT==True:
    df_acc.to_csv(csv_path + '3DVM_seg_atlas_validation_name_AFT_new.csv', index=False)
    df_time.to_csv(csv_path + '3DVM_execute_time_AFT_new.csv', index=False)
    df_jac.to_csv(csv_path + '3DVM_jac_AFT_new.csv', index=False)
    df_acc_subj.to_csv(csv_path + '3DVM_acc_subj_AFT_new.csv', index=False)
