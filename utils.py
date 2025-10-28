import torch
import pywt
import math
import cv2
import numpy as np
from torch import nn
from torch.autograd import Function


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def down_samp(sr, mask):
    sr = torch.fft.fft2(sr, dim=(2, 3))
    sr = torch.fft.fftshift(sr, dim=(2, 3))
    lr = sr*mask
    x_res = torch.fft.ifftshift(lr, dim=(2, 3))
    x_res = torch.abs(torch.fft.ifft2(x_res, dim=(2, 3)))

    return x_res

# def dc(sr, hr, mask):
#     sr = torch.fft.fft2(sr, dim=(2, 3))
#     sr = torch.fft.fftshift(sr, dim=(2, 3))
#     hr = torch.fft.fft2(hr, dim=(2, 3))
#     hr = torch.fft.fftshift(hr, dim=(2, 3))
#     out = sr*(1-mask) + hr*mask
#     x_res = torch.fft.ifftshift(out, dim=(2, 3))
#     x_res = torch.abs(torch.fft.ifft2(x_res, dim=(2, 3)))
#
#     return x_res


def dc(sr, hr, mask):
    sr = torch.fft.fft2(sr, dim=(2, 3))
    sr = torch.fft.fftshift(sr, dim=(2, 3))
    hr = torch.fft.fft2(hr, dim=(2, 3))
    hr = torch.fft.fftshift(hr, dim=(2, 3))
    for i in range(255):
        if mask[0][i] == 1:
            sr[:, :, :, i] = 2*(sr[:, :, :, i] + 0.5 *hr[:, :, :, i]) / 3
    out = sr
    x_res = torch.fft.ifftshift(out, dim=(2, 3))
    x_res = torch.abs(torch.fft.ifft2(x_res, dim=(2, 3)))

    return x_res


# def circleshift(x, shiftnum):
#     c = x.shape[1]    #通道数
#     temp = x[:, 0, :, :]         #先取第0个通道
#     temp = temp.unsqueeze(1)
#     for i in range(1, c):
#         cur = torch.cat((x[:, i, i * shiftnum:, :], x[:, i, :i * shiftnum, :]), 1)            #将第i个通道的行进行circleshift,
#         cur = cur.unsqueeze(1)
#         temp = torch.cat((temp, cur), 1)              #与已经shift好的前几个通道拼接
#
#     return temp


def circleshift(x, shiftnum):
    x = torch.cat((x[:, shiftnum:, :], x[:, : shiftnum, :]), 1)            #将第i个通道的行进行circleshift,
    return x

def haha(x, shiftnum):   # b, c, h, w

    c = x.shape[1]
    for i in range(c):
        x[:, i, :, :] = circleshift(x[:, i, :, :], i*shiftnum)    # b, h, w
    return x

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None


class DWT_2D(nn.Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """
    def __init__(self,wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class LOSS_DWT(nn.Module):
    """
        for ResNet_A
        X --> 1/4*(X_ll + X_lh + X_hl + X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(LOSS_DWT, self).__init__()
        self.dwt = DWT_2D(wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return HH



def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
   # compute 1 dimension gaussian
   gaussian_1D = np.linspace(-1, 1, k)
   # compute a grid distance from center
   x, y = np.meshgrid(gaussian_1D, gaussian_1D)
   distance = (x ** 2 + y ** 2) ** 0.5

   # compute the 2 dimension gaussian
   gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
   gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

   # normalize part (mathematically)
   if normalize:
       gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
   return gaussian_2D


def get_sobel_kernel(k=3):
   # get range
   range = np.linspace(-(k // 2), k // 2, k)
   # compute a grid the numerator and the axis-distances
   x, y = np.meshgrid(range, range)
   sobel_2D_numerator = x
   sobel_2D_denominator = (x ** 2 + y ** 2)
   sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
   sobel_2D = sobel_2D_numerator / sobel_2D_denominator
   return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
   k_thin = 3  # actual size of the directional kernel
   # increase for a while to avoid interpolation when rotating
   k_increased = k_thin + 2

   # get 0° angle directional kernel
   thin_kernel_0 = np.zeros((k_increased, k_increased))
   thin_kernel_0[k_increased // 2, k_increased // 2] = 1
   thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

   # rotate the 0° angle directional kernel to get the other ones
   thin_kernels = []
   for angle in range(start, end, step):
       (h, w) = thin_kernel_0.shape
       # get the center to not rotate around the (0, 0) coord point
       center = (w // 2, h // 2)
       # apply rotation
       rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
       kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

       # get the k=3 kerne
       kernel_angle = kernel_angle_increased[1:-1, 1:-1]
       is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
       kernel_angle = kernel_angle * is_diag  # because of the interpolation
       thin_kernels.append(kernel_angle)
   return thin_kernels


class CannyFilter(nn.Module):
   def __init__(self,
                k_gaussian=3,
                mu=0,
                sigma=1,
                k_sobel=3,
                device = 'cuda:0'):
       super(CannyFilter, self).__init__()
       # device
       self.device = device
       # gaussian
       gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
       self.gaussian_filter = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_gaussian,
                                        padding=k_gaussian // 2,
                                        bias=False)
       self.gaussian_filter.weight.data[:,:] = nn.Parameter(torch.from_numpy(gaussian_2D), requires_grad=False)

       # sobel

       sobel_2D = get_sobel_kernel(k_sobel)
       self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
       self.sobel_filter_x.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D), requires_grad=False)

       self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
       self.sobel_filter_y.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D.T), requires_grad=False)

       # thin

       thin_kernels = get_thin_kernels()
       directional_kernels = np.stack(thin_kernels)

       self.directional_filter = nn.Conv2d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=thin_kernels[0].shape,
                                           padding=thin_kernels[0].shape[-1] // 2,
                                           bias=False)
       self.directional_filter.weight.data[:, 0] = nn.Parameter(torch.from_numpy(directional_kernels), requires_grad=False)

       # hysteresis

       hysteresis = np.ones((3, 3)) + 0.25
       self.hysteresis = nn.Conv2d(in_channels=1,
                                   out_channels=1,
                                   kernel_size=3,
                                   padding=1,
                                   bias=False)
       self.hysteresis.weight.data[:,:] = nn.Parameter(torch.from_numpy(hysteresis), requires_grad=False)

   def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=True):
       # set the setps tensors
       B, C, H, W = img.shape
       blurred = torch.zeros((B, C, H, W)).to(self.device)
       grad_x = torch.zeros((B, 1, H, W)).to(self.device)
       grad_y = torch.zeros((B, 1, H, W)).to(self.device)
       grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
       grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

       # gaussian

       for c in range(C):
           blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])
           grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
           grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

       # thick edges

       grad_x, grad_y = grad_x / C, grad_y / C
       grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
       grad_orientation = torch.atan2(grad_y, grad_x)
       grad_orientation = grad_orientation * (180 / np.pi) + 180  # convert to degree
       grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

       # thin edges

       directional = self.directional_filter(grad_magnitude)
       # get indices of positive and negative directions
       positive_idx = (grad_orientation / 45) % 8
       negative_idx = ((grad_orientation / 45) + 4) % 8
       thin_edges = grad_magnitude.clone()
       # non maximum suppression direction by direction
       for pos_i in range(4):
           neg_i = pos_i + 4
           # get the oriented grad for the angle
           is_oriented_i = (positive_idx == pos_i) * 1
           is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
           pos_directional = directional[:, pos_i]
           neg_directional = directional[:, neg_i]
           selected_direction = torch.stack([pos_directional, neg_directional])

           # get the local maximum pixels for the angle
           # selected_direction.min(dim=0)返回一个列表[0]中包含两者中的小的，[1]包含了小值的索引
           is_max = selected_direction.min(dim=0)[0] > 0.0
           is_max = torch.unsqueeze(is_max, dim=1)

           # apply non maximum suppression
           to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
           thin_edges[to_remove] = 0.0

       # thresholds

       if low_threshold is not None:
           low = thin_edges > low_threshold

           if high_threshold is not None:
               high = thin_edges > high_threshold
               # get black/gray/white only
               thin_edges = low * 0.5 + high * 0.5

               if hysteresis:
                   # get weaks and check if they are high or not
                   weak = (thin_edges == 0.5) * 1
                   weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                   thin_edges = high * 1 + weak_is_high * 1
           else:
               thin_edges = low * 1

       return thin_edges

def canny_loss(hr, sr):
    B, C, H, W = hr.shape
    sum = 0
    count = 0
    for i in range(H):
        for j in range(W):
            if hr[0, 0, i, j] != 0 or sr[0, 0, i, j] != 0:
                sub = torch.abs(hr[0, 0, i, j] - sr[0, 0, i, j])
                sum += sub
                count += 1
    output = sum / count

    return output


