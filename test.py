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
import argparse
from model4 import Net
from module import generate_grid,SpatialTransformNearest
parser = argparse.ArgumentParser(description='param')
parser.add_argument("--gpu", type=str, help="gpu id",dest="gpu", default='1')
parser.add_argument('--atlas_file', default="/home/songlei/OASdata/OAS_new/atlases/", type=str)#UM_379
parser.add_argument('--checkpoint_path',
                    default="/home/newdisk/songlei/train/version_new/ResT_Jac/V3/checkpoint&log/BN_False/lsMSElr5e-05/s0.15/norm_120000_checkpoint.pth.tar",
                    type=str)
parser.add_argument('--test_file', default="/home/songlei/OASdata/OAS_new/valsets/", type=str)
args = parser.parse_args()
def test():
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)
    model = Net((1, 1, 96, 112, 96), 3)
    check_point = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(check_point['state_dict'])
    model = model.to(device)
    model.eval()
    STN_nea = SpatialTransformNearest()
    grid = generate_grid((97, 113, 97))
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda(device).float()
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    '''labels = [2, 4, 8, 10, 11, 12, 13, 16, 17, 18, 24, 26, 28,
              47, 49, 50, 51, 52, 58, 60, ]'''
    val_acc = 0.0
    atlas_paths = glob.glob(os.path.join(args.atlas_file + '*_MR1.npy'))
    val_paths = glob.glob(os.path.join(args.test_file + '*_MR1.npy'))
    JAC = []
    Dice = []
    Time = []
    for atlas_path in atlas_paths:
       atlas_name = os.path.split(atlas_path)[1][:-4]
       atlas = np.load(atlas_path)
       atlas_label = np.load(args.atlas_file+atlas_name+"_label.npy")
       atlas = torch.Tensor(atlas).cuda(device=device).unsqueeze(0).unsqueeze(0)
       for val_path in val_paths:
                val_name = os.path.split(val_path)[1][:-4]
                val = np.load(val_path)
                val_label = np.load(args.test_file + val_name + "_label.npy")

                val=torch.Tensor(val).cuda(device=device).unsqueeze(0).unsqueeze(0)
                val_label=torch.Tensor(val_label).cuda(device=device).unsqueeze(0).unsqueeze(0)

                import time
                time1 = time.time()
                pred, flow, atlas_half, pred_half,X_Y_flow_half = model(val,atlas, grid)
                pred_img = STN(val.shape[2:], val, flow, mode="bilinear")
                time = time.time()-time1
                #print(time)
                Time.append(time)
                flow_per = X_Y_flow_half.permute(0, 2, 3, 4, 1)
                flow_per = flow_per.squeeze(0).detach().cpu()
                #np.save("/home/songlei/flow.npy",flow_per)

                jac = jacobian_determinant1(flow_per*2)
                jac = jac.squeeze()
                jac_neg_per = np.sum([i <= 0 for i in jac]) / (jac.shape[0] * jac.shape[1] * jac.shape[2])
                JAC.append(jac_neg_per)


                pred_label=STN(val_label.shape[2:],val_label,flow)
                # pred_label = STN_nea(val_label, flow.permute(0, 2, 3, 4, 1), grid[:, 0:96, 0:112, 0:96, :])
                pred_label=pred_label.squeeze(0).squeeze(0).detach().cpu().numpy()
                acc=dice(atlas_label,pred_label,labels)
                Dice.append(acc)

                print("dice: ", acc)
                print("Jac:", jac_neg_per,"\n")
    print("DICE:", np.mean(Dice), "std:", np.std(Dice))
    print("JAC:", np.mean(JAC), "std:", np.std(JAC))
    print("Time:", np.mean(Time), "std:", np.std(Time))



def STN(size, src, flow, mode='nearest'):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)  # y, x, z
    grid = torch.unsqueeze(grid, 0)  # add batch
    grid = grid.type(torch.FloatTensor).to(device)

    # print(size)

    new_locs = grid + flow
    # print(new_locs.shape)
    # print("new_locs.mean : {}".format(new_locs.mean()))
    # print("new_locs.shape: {}".format(new_locs.shape))
    shape = flow.shape[2:]

    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * ((new_locs[:, i, ...] / (shape[i] - 1) - 0.5))

    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # 最里边这一维的第一列和第0列的数据
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        # print("new_locs_origin.shape : {}".format(new_locs.shape))
        new_locs = new_locs[..., [2, 1, 0]]
        # print("new_locs.shape : {}".format(new_locs.shape))
        new_locs_construct = new_locs
    #     print("new_locs_construct max : {}  min: {}  mean : {}".format(new_locs_construct.max(),new_locs_construct.min(),new_locs_construct.mean()))
    # print("new_locs' mean: {} ,min: {},max: {}".format(new_locs_construct.mean(),new_locs_construct.min(),new_locs_construct.max()))
    return nnf.grid_sample(src, new_locs_construct, align_corners=True, mode=mode)

def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return np.mean(dicem)
import pystrum.pynd.ndutils as nd
def jacobian_determinant1(disp):
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

    else: # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
if __name__ == "__main__":
    test()































