# 开发时间：16:5916:59
import torch
from torch import nn

from MambaMRI.data.subsample import create_mask_for_mask_type


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock_E(nn.Module):
    def __init__(self, kernel_size=3, dims=16):
        super(BasicBlock_E, self).__init__()
        self.kernel_size = kernel_size
        self.dims = dims
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')
        block_1 = [
            nn.Sequential(
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True),
                # nn.LeakyReLU(inplace=True)
            )
        ]
        self.block_1 = nn.Sequential(*block_1)
        block_2 = [
            nn.Sequential(
                nn.Conv2d(dims, dims // 4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True),
                nn.Conv2d(dims // 4, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            )
        ]
        self.block_2 = nn.Sequential(*block_2)

    def forward(self, x):
        lp = self.block_1(self.upsample(self.avgpool(x)))
        hp = self.block_2(x - lp)
        x = lp + hp

        return x


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x_fused):
        x_fused = self.down(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused)

        return x_out

class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x):
        x_fused = self.fusion_conv(x)
        return x_fused


class DC(nn.Module):
    def __init__(self):
        super(DC, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)

    def forward(self, generated, masked_imag_ks):
        mask = create_mask_for_mask_type('random', [0.08], [4])
        mask_shape = (256,256, 1)  # 第一维度通常是通道数，这里假设为单通道
        mask = mask(mask_shape, seed=1234)
        mask = mask.repeat(1, 1, 256).permute(0,2,1).unsqueeze(0).to(torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
        gene_fft = torch.fft.fftshift(torch.fft.fft2(generated, dim=(2, 3)), dim=(2, 3))
        result = mask * (gene_fft * self.w / (1 + self.w) + masked_imag_ks * 1 / (self.w + 1))
        out_fft = result + (1 - mask) * gene_fft
        # plt.subplot(221)
        # plt.plot(abs(masked_imag_ks[0,0,:,:].T.detach().cpu().numpy()))
        # plt.subplot(222)
        # plt.plot(abs(gene_fft[0,0,:,:].T.detach().cpu().numpy()))
        # plt.subplot(223)
        # plt.plot(abs(result[0,0,:,:].T.detach().cpu().numpy()))
        # plt.subplot(224)
        # plt.plot(abs(out_fft[0,0,:,:].T.detach().cpu().numpy()))
        # plt.show()
        output_complex = torch.fft.ifft2(torch.fft.ifftshift(out_fft, dim=(2, 3)), dim=(2, 3))
        output = abs(output_complex)
        # plt.imshow(output[0,0,:,:].detach().cpu().numpy())
        # plt.show()
        return output

