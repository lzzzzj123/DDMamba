import argparse
import os
import copy

import torch
from MambaMRI import pytorch_ssim
from torch import nn
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from MambaMRI.MambaMRI_structure.models.MID_net import IDNet as Model
from MambaMRI.datasets import TrainDataset, EvalDataset
from MambaMRI.utils import AverageMeter, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='/media/amax/2A20E8CB20E89ED7/DDMamba/Datasets/NAMIC/train_randsample_4.h5')
    parser.add_argument('--eval-file', type=str, default='/media/amax/2A20E8CB20E89ED7/DDMamba/Datasets/NAMIC/val_randsample_4.h5')
    # parser.add_argument('--train-file', type=str,
    #                     default='/media/amax/2A20E8CB20E89ED7/Lzj/Datasets/NAMIC/train_randsample_8.h5')
    # parser.add_argument('--eval-file', type=str,
    #                     default='/media/amax/2A20E8CB20E89ED7/Lzj/Datasets/NAMIC/val_randsample_8.h5')

    parser.add_argument('--outputs-dir', type=str, default='results/NAMIC')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--gpu', type=str, default=[0,1], help='GPUs')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    ######### Set GPUs ###########
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    ######## DataParallel ###########
    model = Model()
    model = model.to(device)
    # if len(args.gpu) > 1:
    #     model = torch.nn.DataParallel(model, args.gpu)

    criterion = nn.MSELoss()
    # ssim_loss = pytorch_ssim.SSIM()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))


    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    best_weights = copy.deepcopy(model.state_dict())

    best_epoch = 0
    best_psnr = 0.0
    LOSS = []
    PSNR = []
    SSIM = []
    # input = torch.randn(1, 2, 240, 240).to(device)
    # flops, params = profile(model, (input,))
    # print('flops:', flops, 'params:', params)
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))
            for data in train_dataloader:
                inputs, pd, labels = data
                inputs = inputs.to(device)
                pd = pd.to(device)
                labels = labels.to(device)
                preds = model(inputs, pd)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        LOSS.append(epoch_losses.avg)
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        ssim = 0
        for data in eval_dataloader:
            inputs, pd, labels = data
            inputs = inputs.to(device)
            pd = pd.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(inputs, pd)
                preds = preds.clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            ssim += pytorch_ssim.ssim(preds, labels)
        ssim = ssim/len(eval_dataset)
        SSIM.append(ssim.cpu().numpy())
        PSNR.append(epoch_psnr.avg.cpu().numpy())
        print('eval psnr: {:.2f}    eval ssim: {:.4f}'.format(epoch_psnr.avg, ssim))
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
    np.save('results/NAMIC/LOSS_4.npy', LOSS)
    np.save('results/NAMIC/PSNR_4.npy', PSNR)
    np.save('results/NAMIC/SSIM_4.npy', SSIM)
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
