# 开发时间：10:5210:52
import torch

import torch.nn as nn
from matplotlib import pyplot as plt

from MambaMRI.mambair_arch import MambaIR


class ks_net_block0(nn.Module):
    def __init__(self):
        super(ks_net_block0, self).__init__()
        self.mamba = MambaIR(img_size=256,
                           patch_size=1,
                           in_chans=2,
                           embed_dim=16,
                           depths=([1, 2]),
                           mlp_ratio=2.,
                           drop_rate=0.,
                           d_state=16,
                           norm_layer=nn.LayerNorm,
                           patch_norm=True,
                           use_checkpoint=False,
                           upscale=1,
                           img_range=1.,
                           upsampler='pixelshuffle',
                           resi_connection='1conv')
        self.conv1 = nn.Conv2d(2,1,kernel_size=3,stride=1,padding='same')

    def forward(self, x, PD):
        x = torch.cat([x,PD],dim=1)
        x_k = self.mamba(x)
        # x_k = self.conv1(x_k)
        # x_k = F.silu(x_k)
        x = abs(x_k)
        return x
class ks_net_block1(nn.Module):
    def __init__(self):
        super(ks_net_block1, self).__init__()
        self.mamba = MambaIR(img_size=256,
                           patch_size=1,
                           in_chans=17,
                           embed_dim=16,
                           depths=([1, 2]),
                           mlp_ratio=2.,
                           drop_rate=0.,
                           d_state=16,
                           norm_layer=nn.LayerNorm,
                           patch_norm=True,
                           use_checkpoint=False,
                           upscale=1,
                           img_range=1.,
                           upsampler='pixelshuffle',
                           resi_connection='1conv')
        self.conv1 = nn.Conv2d(2,1,kernel_size=3,stride=1,padding='same')

    def forward(self, x):
        x_k = self.mamba(x)
        # x_k = self.conv1(x_k)
        # x_k = F.silu(x_k)
        x = abs(x_k)
        return x
class ks_net_block2(nn.Module):
    def __init__(self):
        super(ks_net_block2, self).__init__()
        self.mamba = MambaIR(img_size=256,
                           patch_size=1,
                           in_chans=33,
                           embed_dim=16,
                           depths=([1, 2]),
                           mlp_ratio=2.,
                           drop_rate=0.,
                           d_state=16,
                           norm_layer=nn.LayerNorm,
                           patch_norm=True,
                           use_checkpoint=False,
                           upscale=1,
                           img_range=1.,
                           upsampler='pixelshuffle',
                           resi_connection='1conv')
        self.conv1 = nn.Conv2d(3,1,kernel_size=3,stride=1,padding='same')

    def forward(self, x):
        x_k = self.mamba(x)
        # x_k = self.conv1(x_k)
        # x_k = F.silu(x_k)
        x = abs(x_k)
        return x
class ks_net_block3(nn.Module):
    def __init__(self):
        super(ks_net_block3, self).__init__()
        self.mamba = MambaIR(img_size=256,
                           patch_size=1,
                           in_chans=49,
                           embed_dim=16,
                           depths=([1, 2]),
                           mlp_ratio=2.,
                           drop_rate=0.,
                           d_state=16,
                           norm_layer=nn.LayerNorm,
                           patch_norm=True,
                           use_checkpoint=False,
                           upscale=1,
                           img_range=1.,
                           upsampler='pixelshuffle',
                           resi_connection='1conv')
        self.conv1 = nn.Conv2d(4,1,kernel_size=3,stride=1,padding='same')

    def forward(self, x):
        x_k = self.mamba(x)
        # x_k = self.conv1(x_k)
        # x_k = F.silu(x_k)
        x = abs(x_k)
        return x
class ks_net_block4(nn.Module):
    def __init__(self):
        super(ks_net_block4, self).__init__()
        self.mamba = MambaIR(img_size=256,
                           patch_size=1,
                           in_chans=65,
                           embed_dim=16,
                           depths=([1, 2]),
                           mlp_ratio=2.,
                           drop_rate=0.,
                           d_state=16,
                           norm_layer=nn.LayerNorm,
                           patch_norm=True,
                           use_checkpoint=False,
                           upscale=1,
                           img_range=1.,
                           upsampler='pixelshuffle',
                           resi_connection='1conv')
        self.conv1 = nn.Conv2d(5,1,kernel_size=3,stride=1,padding='same')

    def forward(self, x):
        x_k = self.mamba(x)
        # x_k = self.conv1(x_k)
        # x_k = F.silu(x_k)
        x = abs(x_k)
        return x


class IDNet(nn.Module):
    def __init__(self):
        super(IDNet, self).__init__()
        self.ks_block1 = ks_net_block0()
        self.ks_block2 = ks_net_block1()
        self.ks_block3 = ks_net_block2()
        self.ks_block4 = ks_net_block3()
        self.ks_block5 = ks_net_block4()
        self.conv0 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding='same')
        self.conv1 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding='same')
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding='same')
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding='same')
        self.conv4 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding='same')
        self.conv5 = nn.Conv2d(80, 1, kernel_size=1, stride=1, padding='same')
        self.conv_last = nn.Sequential(
            nn.Conv2d(81, 16, 1, 1, 'same'),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, 1, 'same'))


    def forward(self, X, PD):
        temp = X
        # PD = X
        k_1 = self.ks_block1(temp, PD)
        k_1 = torch.cat([k_1, temp], dim=1)
        k_2 = self.ks_block2(k_1)
        k_2 = torch.cat([k_2, k_1], dim=1)
        k_3 = self.ks_block3(k_2)
        k_3 = torch.cat([k_3, k_2], dim=1)
        k_4 = self.ks_block4(k_3)
        k_4 = torch.cat([k_4, k_3], dim=1)
        k_5 = self.ks_block5(k_4)
        plt.show()
        k_5 = torch.cat([k_5, k_4], dim=1)
        k_5 = self.conv_last(k_5)

        temp = k_5
        return temp