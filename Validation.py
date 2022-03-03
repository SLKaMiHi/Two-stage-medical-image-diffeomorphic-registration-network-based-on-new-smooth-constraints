import torch
import time
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import colors
from losses import jacobian_determinant,Get_Jac,jacobian_determinant1
import torch
import torch.nn.functional as nnf
import os
from module import SpatialTransformNearest

device1 = torch.device('cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu')
def validation(grid, atlas_file, val_file,acc_fn,model,labels, mode='whole',slice=None,label_path=None,show_img='OASIS_OAS1_0300_MR1', show_atlas='OASIS_OAS1_0222_MR1'):
    """
    show the 'S02' img registration during training.
    :param val_data:
    :param targ:
    :param acc_fn:
    :param model:
    :param mode:
    :param sclice: select a slice show
    :param label_path:
    :param writer: tensorboardX writer
    :param img: image name to show
    :return: average validation accuracy, summary time spend.
    """


    model.eval()
    val_acc=0.0
    atlas_paths=glob.glob(os.path.join(atlas_file+'*_MR1.npy'))
    val_paths = glob.glob(os.path.join(val_file+'*_MR1.npy'))
    STN = SpatialTransformNearest().to(device1)
    # print(model)
    start_time = time.time()
    JAC = []
    Dice = []
    for atlas_path in atlas_paths:
       atlas_name = os.path.split(atlas_path)[1][:-4]
       atlas = np.load(atlas_path)
       atlas_label = np.load(atlas_file+atlas_name+"_label.npy")
       atlas = torch.Tensor(atlas).cuda(device=device1).unsqueeze(0).unsqueeze(0)
       for val_path in val_paths:
                val_name = os.path.split(val_path)[1][:-4]
                val = np.load(val_path)
                val_label = np.load(val_file + val_name + "_label.npy")

                val=torch.Tensor(val).cuda(device=device1).unsqueeze(0).unsqueeze(0)
                val_label=torch.Tensor(val_label).cuda(device=device1).unsqueeze(0).unsqueeze(0)

                pred, flow, atlas_half, pred_half, flow_half = model(val,atlas, grid)

                flow_per = flow.permute(0, 2, 3, 4, 1)
                flow_per = flow_per.squeeze(0).detach().cpu()
                #jac = jacobian_determinant(flow_per)
                jac = jacobian_determinant1(flow_per)
                jac = jac.squeeze()
                jac_neg_per = np.sum([i < 0 for i in jac]) / (jac.shape[0] * jac.shape[1] * jac.shape[2])
                JAC.append(jac_neg_per)


                pred_label=STN(val_label,flow.permute(0, 2, 3, 4, 1), grid[:, 0:96, 0:112, 0:96, :])
                pred_label=pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()
                acc=acc_fn(atlas_label,pred_label,labels)
                Dice.append(acc)
                # print(volume_path)
                if val_name == show_img and atlas_name == show_atlas:
                    atlas_slice=atlas[0,0,:,slice,:]
                    atlas_slice=atlas_slice.squeeze(0).squeeze(0).detach().cpu().numpy()
                    val_slice=val[0,0,:,slice,:]
                    val_slice=val_slice.squeeze(0).squeeze(0).detach().cpu().numpy()

                    pred_slice=pred[0,0,:,slice,:]
                    pred_slice=pred_slice.squeeze(0).squeeze(0).detach().cpu().numpy()

                    # print("flow.shape:{}".format(flow.shape))
                    flow_per=flow.permute(0,2,3,4,1)
                    flow_per = flow_per.squeeze(0).detach().cpu()
                    jac_det=jacobian_determinant1(flow_per)
                    # print("jac_det.shape:{}".format(jac_det.shape))
                    jac_det=jac_det.squeeze()
                    jac_det_slice=jac_det[:,slice,:]
                    jac_det_slice=jac_det_slice

                    #label
                    atlas_label_slice = atlas_label[:,slice,:]
                    pred_label_slice = pred_label
                    pred_label_slice = pred_label_slice[:,slice,:]

    time_spend=time.time()-start_time

    return np.mean(Dice),time_spend,atlas_slice,val_slice,pred_slice,jac_det_slice,flow,np.mean(JAC),atlas_label_slice,pred_label_slice




def show(atlas,img,pred,jac_det):
    num=5
    fig, ax = plt.subplots(1,num)
    fig.dpi=150

    ax0=ax[0].imshow(atlas,cmap='gray')
    fig.colorbar(ax0,ax=ax[0],shrink=0.3)

    ax1=ax[1].imshow(img,cmap='gray')
    fig.colorbar(ax1,ax=ax[1],shrink=0.3)

    ax2=ax[2].imshow(pred,cmap='gray')
    fig.colorbar(ax2,ax=ax[2],shrink=0.3)

    ax3=ax[3].imshow(jac_det,cmap='bwr',norm=MidpointNormalize(midpoint=1))
    fig.colorbar(ax3,ax=ax[3],shrink=0.3)

    return fig


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def HistNorm(img,low_bound=0,high_bound=255):
    # used in torch
    img_max=img.max()
    img_min=img.min()
    out=(high_bound-low_bound)/(img_max-img_min)*(img-img_min)+low_bound
    out[out<low_bound]=low_bound
    out[out>high_bound]=high_bound
    return out


def MinMaxNorm(img, use_gpu=False):
    if use_gpu:
        Max = torch.max(img)
        Min = torch.min(img)
        return (img - Min) / (Max - Min)
    else:
        Max = img.max()
        Min = img.min()
        return (img - Min) / (Max - Min)
