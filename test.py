import argparse

import matplotlib.pyplot as plt
import torch
import  os
import numpy as np
import PIL.Image as pil_image
from thop import profile
import torch.nn.functional as F
from MambaMRI import pytorch_ssim
import time
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from MambaMRI.MambaMRI_structure.models.MID_net import IDNet as Model
from MambaMRI.datasets import TestDataset
from MambaMRI.utils import AverageMeter, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='results/NAMIC/x4/best.pth')
    parser.add_argument('--test-file', type=str, default='/media/amax/2A20E8CB20E89ED7/DDMamba/Datasets/NAMIC/test_randsample_4.h5')
    # parser.add_argument('--test-file', type=str, default='/media/amax/新加卷/fastMRI/val.h5')

    # parser.add_argument('--test-file', type=str,
    #                     default='/media/amax/2A20E8CB20E89ED7/Lzj/Mamba_image_base/MambaMRI/Datasets/NAMIC/test_randsample_8.h5')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model().to(device)

    test_dataset = TestDataset(args.test_file)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)


    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()

    psnr = 0
    ssim = 0
    psnr_max = 0
    ssim_max = 0
    psnr_min = 45
    ssim_min = 1
    PSNR = []
    SSIM = []
    PSNR_result = []
    SSIM_result = []
    start = time.time()
    for i, data in enumerate(test_dataloader):
        inputs, pd, labels = data
        # inputs = F.interpolate(inputs, size=(256, 256), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
        # pd = F.interpolate(pd, size=(256, 256), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
        # labels = F.interpolate(labels, size=(256, 256), mode='bicubic', align_corners=True).clamp(0.0, 1.0)

        inputs = inputs.to(device).float()
        pd = pd.to(device)
        labels = labels.to(device).float()
        with torch.no_grad():
            preds = model(inputs, pd).clamp(0.0, 1.0)

        # plt.subplot(221)
        # plt.imshow(inputs[0, 0, :, :].cpu().numpy())
        # plt.subplot(222)
        # plt.imshow(labels[0, 0, :, :].cpu().numpy())
        # plt.subplot(223)
        # plt.imshow(preds[0, 0, :, :].cpu().numpy())
        # plt.subplot(224)
        # plt.imshow(pd[0, 0, :, :].cpu().numpy())
        # plt.show()
        psnr_ = calc_psnr(preds, labels)
        if psnr_ > psnr_max:
            psnr_max = psnr_
        if psnr_ < psnr_min:
            psnr_min = psnr_
        ssim_ = pytorch_ssim.ssim(preds, labels)
        if ssim_ > ssim_max:
            ssim_max = ssim_
        if ssim_ < ssim_min:
            ssim_min = ssim_
        print('i: ', i)
        print('psnr: ', psnr_)
        print('ssim: ', ssim_)
        psnr += psnr_
        ssim += ssim_
        PSNR_result.append(psnr_.cpu().numpy())
        SSIM_result.append(ssim_.cpu().numpy())
        PSNR.append(psnr_)
        SSIM.append(ssim_)


        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array(preds)
        output = output.astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(os.path.join('output/NAMIC/x4', '{}.png'.format(str(i))))
        # labels = labels.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        # labels = np.array(labels)
        # labels = labels.astype(np.uint8)
        # labels = pil_image.fromarray(labels)
        # labels.save(os.path.join('output/NAMIC/labels', '{}.png'.format(str(i))))
    end = time.time()
    dis = end - start
    psnr = psnr / len(test_dataset)
    ssim = ssim / len(test_dataset)
    # np.save('output/NAMIC/PSNR4.npy', PSNR_result)
    # np.save('output/NAMIC/SSIM4.npy', SSIM_result)

    psnr_std = torch.std(torch.stack(PSNR))
    ssim_std = torch.std(torch.stack(SSIM))

    print('eval psnr: {:.2f}    max psnr: {:.2f}   min psnr: {:.2f}'.format(psnr, psnr_max, psnr_min))
    print('eval ssim: {:.4f}    max ssim: {:.4f}   min ssim: {:.4f}'.format(ssim, ssim_max, ssim_min))
    print('psnr_std: ', psnr_std)
    print('ssim_std: ', ssim_std)
    print('dis: ', dis)



