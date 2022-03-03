import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from module import JacboianDet, SpatialTransform

device1 = torch.device('cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu')
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    Modify by me:
         How I can use it for 3D convolution.
    """

    def __init__(self, dim, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.dim = dim
        self.conv1 = conv_block(dim, channel, channel, bn=False)
        self.conv2 = conv_block(dim, channel, channel, bn=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return y*x


#ResT
class SepConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv3d, self).__init__()
        self.depthwise = torch.nn.Conv3d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm3d(in_channels)
        self.pointwise = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 image_size,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False,
                 kernel_size=3,
                 q_stride=1):
        super().__init__()
        self.num_heads = num_heads
        self.img_size = image_size
        head_dim = dim // num_heads
        pad = (kernel_size - q_stride) // 2
        inner_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv3d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm3d(self.num_heads)

    def forward(self, x, H, W, L):
        B, N, C = x.shape
        b, n, _, h = *x.shape, self.num_heads
        xq = rearrange(x, 'b (l w d) n -> b n l w d', l=self.img_size[0], w=self.img_size[1], d=self.img_size[2])
        q = self.q(xq)
        q = rearrange(q, 'b (h d) l w k -> b h (l w k) d', h=h)


        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W, L)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, image_size, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, image_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, L):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, L))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class GL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)




class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))




class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, bn, mode='maintain'):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        self.bn = bn
        # lightweight method choose.
        self.method = 'multi_conv'

        if mode == 'half':
            stride = 2
        elif mode == 'maintain':
            stride = 1
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn=nn.BatchNorm3d(out_channels)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, out):
        """
        Pass the input through the conv_block
            """
        out = self.main(out)
        if self.bn:
            out = self.bn(out)
            print("bn is on")
        out = self.activation(out)
        return out
class Net_half(nn.Module):
    def __init__(self, img_dim,bn=False):

        super(Net_half, self).__init__()
        self.entry_net = conv_block(img_dim, 2, 16, bn)
        self.down1 = conv_block(img_dim, 16, 32, bn, 'half')
        self.down2 = conv_block(img_dim, 32, 64, bn, 'half')
        self.down3 = conv_block(img_dim, 64, 128, bn, 'half')



        #decoder
        self.conv1 = conv_block(img_dim, 128, 64, bn, 'maintain')
        self.upsample_layers11 = conv_block(img_dim, 128, 64, bn, 'maintain')
        self.upsample_layers12 = conv_block(img_dim, 64, 32, bn, 'maintain')
        self.upsample_layers21 = conv_block(img_dim, 64, 32, bn, 'maintain')
        self.upsample_layers22 = conv_block(img_dim, 64, 32, bn, 'maintain')
        self.upsample_layers23 = conv_block(img_dim, 32, 16, bn, 'maintain')
        self.upsample_layers31 = conv_block(img_dim, 32, 16, bn, 'maintain')
        self.upsample_layers32 = conv_block(img_dim, 16, 8, bn, 'maintain')


        #flow
        conv_fn = getattr(nn, f'Conv{img_dim}d')
        self.flow = conv_fn(8, img_dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransform()



        # self.attention = eca_layer(dim)
        self.upsample=nn.Upsample(scale_factor=2)

    def forward(self, src, tgt, grid):
        x = torch.cat([src, tgt], 1)
        x_e1 = self.entry_net(x)
        x_e2 = self.down1(x_e1)
        x_e3 = self.down2(x_e2)
        x_e4 = self.down3(x_e3)

        x_e4 = self.conv1(x_e4)

        #decoder
        x_d3 = self.upsample(x_e4)
        x_d3 = torch.cat([x_d3, x_e3], 1)
        x_d3 = self.upsample_layers11(x_d3)
        x_d3 = self.upsample_layers12(x_d3)

        x_d2 = self.upsample(x_d3)
        x_d2 = torch.cat([x_d2, x_e2], 1)
        x_d2 = self.upsample_layers21(x_d2)
        x_d2 = torch.cat([x_d2, x_e2], 1)
        x_d2 = self.upsample_layers22(x_d2)
        x_d2 = self.upsample_layers23(x_d2)

        x_d1 = self.upsample(x_d2)
        x_d1 = torch.cat([x_d1, x_e1], 1)
        x_d1 = self.upsample_layers31(x_d1)
        x_d1 = self.upsample_layers32(x_d1)

        #x_flow = self.upsample_layers5(x_d1)
        flow = self.flow(x_d1)



        y = self.spatial_transform(src, flow.permute(0, 2, 3, 4, 1), grid)

        return y, flow,x_e1,x_e2,x_e3
class Net(nn.Module):
    def __init__(self,img_dim,bn=False):

        super(Net, self).__init__()
        self.Net_half = Net_half(3)


        self.entry_net = conv_block(img_dim, 2, 16, bn)
        self.down1 = conv_block(img_dim, 16, 32, bn, 'half')
        self.down2 = conv_block(img_dim, 32, 64, bn, 'half')
        self.down3 = conv_block(img_dim, 64, 128, bn, 'half')

        # decoder
        self.conv1 = conv_block(img_dim, 128, 64, bn, 'maintain')
        self.upsample_layers11 = conv_block(img_dim, 192, 64, bn, 'maintain')
        self.upsample_layers12 = conv_block(img_dim, 64, 32, bn, 'maintain')
        self.upsample_layers21 = conv_block(img_dim, 96, 32, bn, 'maintain')
        self.upsample_layers22 = conv_block(img_dim, 96, 32, bn, 'maintain')
        self.upsample_layers23 = conv_block(img_dim, 32, 16, bn, 'maintain')
        self.upsample_layers31 = conv_block(img_dim, 48, 16, bn, 'maintain')
        self.upsample_layers32 = conv_block(img_dim, 16, 8, bn, 'maintain')

        # flow
        conv_fn = getattr(nn, f'Conv{img_dim}d')
        self.flow = conv_fn(8, img_dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransform()

        # self.attention = eca_layer(dim)
        self.upsample = nn.Upsample(scale_factor=2)


        self.upsample=nn.Upsample(scale_factor=2)

    def forward(self, src, tgt, grid):

        src_half = nn.functional.interpolate(src, (48, 56, 48), mode='trilinear')
        tgt_half = nn.functional.interpolate(tgt, (48, 56, 48), mode='trilinear')

        pred_half, flow_half,x_half1,x_half2,x_half3 = self.Net_half(src_half, tgt_half, grid[:, 0:48, 0:56, 0:48, :])
        x_half1 = self.upsample(x_half1)
        x_half2 = self.upsample(x_half2)
        x_half3 = self.upsample(x_half3)
        flow_half_up = nn.functional.interpolate(flow_half, (96, 112, 96), mode='trilinear')

        src_new = self.spatial_transform(src, flow_half_up.permute(0, 2, 3, 4, 1) * 2, grid[:, 0:96, 0:112, 0:96, :])


        x = torch.cat([src_new, tgt], 1)
        x_e1 = self.entry_net(x)
        x_e2 = self.down1(x_e1)
        x_e3 = self.down2(x_e2)
        x_e4 = self.down3(x_e3)


        x_e4 = self.conv1(x_e4)

        #decoder

        #3
        x_d3 = self.upsample(x_e4)


        x_d3 = torch.cat([x_d3, x_half3, x_e3], 1)

        x_d3 = self.upsample_layers11(x_d3)
        x_d3 = self.upsample_layers12(x_d3)

        #2
        x_d2 = self.upsample(x_d3)



        x_d2 = torch.cat([x_d2, x_half2, x_e2], 1)
        x_d2 = self.upsample_layers21(x_d2)
        x_d2 = torch.cat([x_d2, x_half2, x_e2], 1)
        x_d2 = self.upsample_layers22(x_d2)
        x_d2 = self.upsample_layers23(x_d2)

        #1
        x_d1 = self.upsample(x_d2)


        x_d1 = torch.cat([x_d1, x_half1, x_e1], 1)

        x_d1 = self.upsample_layers31(x_d1)
        x_flow = self.upsample_layers32(x_d1)

        flow = self.flow(x_flow)

        flow = flow + flow_half_up * 2


        y = self.spatial_transform(src, flow.permute(0, 2, 3, 4, 1) , grid[:, 0:96, 0:112, 0:96, :])

        return y, flow, tgt_half, pred_half, flow_half_up



