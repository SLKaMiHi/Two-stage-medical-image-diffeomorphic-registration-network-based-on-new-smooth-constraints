import numpy as np
import SimpleITK as sit
import torch
# from module import SpatialTransformNearest,SpatialTransform,generate_grid
from losses import dice
import torch.nn as nn
import torch.nn.functional as nnf


labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
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





flow = sit.GetArrayFromImage(sit.ReadImage("/home/mamingrui/songlei/flow/flow_Syn.nii.gz"))

atlas_label = sit.GetArrayFromImage(sit.ReadImage("/home/mamingrui/songlei/flow/OASIS_OAS1_0456_MR1_label.nii.gz"))
atlas = sit.GetArrayFromImage(sit.ReadImage("/home/mamingrui/songlei/flow/OASIS_OAS1_0456_MR1.nii.gz"))
val_label = sit.GetArrayFromImage(sit.ReadImage("/home/mamingrui/songlei/flow/OASIS_OAS1_0434_MR1_label.nii.gz"))
val = sit.GetArrayFromImage(sit.ReadImage("/home/mamingrui/songlei/flow/OASIS_OAS1_0434_MR1.nii.gz"))
# grid = generate_grid((225, 193, 161))
# grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

flow = torch.from_numpy(flow).unsqueeze(0).cuda()
print(flow.shape)
flow = flow.permute(0,4,1,2,3)
# atlas= torch.from_numpy(atlas).unsqueeze(0).unsqueeze(0).cuda()
val = torch.from_numpy(val).unsqueeze(0).unsqueeze(0).cuda().float()
val_label = torch.from_numpy(val_label).unsqueeze(0).unsqueeze(0).cuda().float()

# STN = SpatialTransform()
# Stn = SpatialTransformNearest()

STN = SpatialTransformer([224, 192, 160], 'nearest').cuda()
stn = SpatialTransformer([224, 192, 160], 'bilinear').cuda()

moved_label = STN(val_label, flow).data.cpu().numpy()[0, 0, :, :, :]
moved = stn(val, flow).data.cpu().numpy()[0, 0, :, :, :]

sit.WriteImage(sit.GetImageFromArray(moved_label), "/home/mamingrui/songlei/flow/Moved_Syn_label.nii.gz")
sit.WriteImage(sit.GetImageFromArray(moved), "/home/mamingrui/songlei/flow/Moved_Syn.nii.gz")
# sit.WriteImage(sit.GetImageFromArray(flow.data.cpu().numpy()[0, :, :, :, :]*100), "/home/mamingrui/songlei/flow/flow_SYM.nii.gz")

# sit.WriteImage(sit.GetImageFromArray(atlas), "/home/mamingrui/songlei/flow/atlas.nii.gz")
# sit.WriteImage(sit.GetImageFromArray(atlas_label), "/home/mamingrui/songlei/flow/atlas_label.nii.gz")
# sit.WriteImage(sit.GetImageFromArray(val.data.cpu().numpy()[0, 0, :, :, :]), "/home/mamingrui/songlei/flow/Moving.nii.gz")
# sit.WriteImage(sit.GetImageFromArray(val_label.data.cpu().numpy()[0, 0, :, :, :]), "/home/mamingrui/songlei/flow/Moving_label.nii.gz")
dic = dice(moved_label,atlas_label,labels)
print(dic)







