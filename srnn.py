import torch
from torch import distributed, optim
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from utils import *
from datetime import datetime
import os
import json
import numpy as np
import cv2
import torch.nn.functional as F

from skimage.measure import compare_psnr, compare_ssim
from DRAN import DRAN
from model.srcnn import Net
# from model.fsrcnn import Net
from model.SRMDNF import SRMD

# from loss import *
import math
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = range(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SRNN(object):

    def __init__(self, args, trainable):
        """初始化模块"""
        self.p = args
        self.trainable = trainable
        self._compile()


    def _compile(self):
        """编译模块（神经网络结构，损失函数，优化器等）"""
        # Model
        self.model = DRAN(self.p)
        # loss functiong
        self.content_loss = nn.L1Loss()

        self.start_epoch = 1
        # optimizer and loss
        if self.trainable:
            # self.loss = self.loss.to(device)
            self.content_loss = self.content_loss.to(device)
            self.optim = Adam(self.model.parameters(), lr=self.p.lr)
            # adjust earning rate
            self.scheduler = lr_scheduler.StepLR(optimizer=self.optim, step_size=300, gamma=0.5, last_epoch=-1)

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.to(device)
            if len(device_ids) > 1:
                print('dataparallel')
                self.model = nn.DataParallel(self.model, device_ids=device_ids)


        if self.p.load_ckpt and self.trainable:
            print('Loading checkpoint from{}\n'.format(self.p.ckpt_file))
            # original saved file with DataParallel
            state_dict = torch.load(self.p.ckpt_file)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                # name = k[7:]  # remove `module.`
                # new_state_dict[name] = v
                if 'module.' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
            # load params
            self.model.load_state_dict(new_state_dict)
            self.start_epoch = state_dict['epoch'] + 1 # epoch
            self.optim.load_state_dict(state_dict['optimizer'])
            for state in self.optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            self.optim.param_groups[0]['lr'] = self.p.lr
            # self.scheduler.load_state_dict(state_dict['lr_scheduler'])
            print("current epoch: {}".format(self.start_epoch))

    def save_model(self, epoch, best_psnr, best_epoch, stats):
        # checkpiont and stats
        if not(self.p.load_ckpt) and epoch<10:
            localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            save_dir_name = 'DRAN' + localtime
            self.ckpt_dir = os.path.join(self.p.ckpt_save_dir, save_dir_name)
            self.stats_dir = os.path.join(self.p.stats_save_dir, save_dir_name)
            if not os.path.isdir(self.p.ckpt_save_dir):
                os.mkdir(self.p.ckpt_save_dir)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)
            if not os.path.isdir(self.p.stats_save_dir):
                os.mkdir(self.p.loss_save_dir)
            if not os.path.isdir(self.stats_dir):
                os.mkdir(self.stats_dir)
            save_args(self.p, self.stats_dir)

        if self.p.load_ckpt:
            save_dir_name = self.p.ckpt_file.split('/')[3]
            self.ckpt_dir = os.path.dirname(self.p.ckpt_file)
            self.stats_dir = os.path.join(self.p.stats_save_dir, save_dir_name)
            # save
            save_args(self.p, self.stats_dir)
        # checkpoint
        if self.p.ckpt_overwrite:
            filename = 'latest.pt'
        else:
            psnr = stats['valid_psnr'][epoch-self.start_epoch]
            # valid_loss = stats['valid_loss'][epoch-]
            filename = 'epoch{}-{:.4f}.pt'.format(epoch, psnr)

        filedir = '{}/{}'.format(self.ckpt_dir, filename)
        print('Saving checkpoint to : {}\n'.format(filename))
        torch.save({'model': self.model.module.state_dict(), 'optimizer': self.optim.state_dict(), 'epoch': epoch, 'lr_scheduler': self.scheduler.state_dict()}, filedir)

        if epoch == best_epoch:
            torch.save(self.model.state_dict(), '{}/best.pt'.format(self.ckpt_dir))
            with open('{}/best_epoch.txt'.format(self.stats_dir), 'w') as f:
                f.write('Best: {:.2f} @epoch {}'.format(best_psnr, best_epoch))

        # stats(保存loss等状态信息)保存为JSON
        stats_dict = '{}/stats.json'.format(self.stats_dir)
        with open(stats_dict, 'w') as sd:
            json.dump(stats, sd, indent=2)

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).to(device)
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        return torch.cat((mixed_x, x), 0), torch.cat((mixed_y, y), 0)

    def train(self, train_loader, valid_loader):
        '''Train on training set.'''
        self.model.train(True)

        num_batches = len(train_loader)

        # Dictionaries of tracked stats
        stats = {'learning_rate': [],
                 'train_loss': [],
                 'loss': [],
                 'valid_psnr': [],
                 'valid_ssim': []}

        # Main training loop
        train_start = datetime.now()
        epoch_save = 5

        for _epoch in range(self.start_epoch, self.p.epochs+1):
            epoch_start = datetime.now()
            print('EPOCH : {}/{}'.format(_epoch, self.p.epochs))

            train_loss_tracker = Tracker()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):

                progress_bar(batch_idx, num_batches, train_loss_tracker.avg)

                if self.use_cuda:
                    source = source.to(device)
                    target = target.to(device)
                # print(source.shape, target.shape)
                # source, target = self.mixup_data(source, target)
                # source = source.type(torch.cuda.DoubleTensor)
                # target = target.type(torch.cuda.DoubleTensor)

                source_SR = self.model(source)
                #loss
                content_loss = self.content_loss(source_SR, target)
                # perceptual_loss = self.perceptual_loss(source_SR, target)
                # loss = self.content_loss_factor * content_loss + self.perceptual_loss_factor * perceptual_loss
                loss = content_loss
                train_loss_tracker.update(loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            self.scheduler.step()
            epoch_end = datetime.now()
            # epoch end

            stats['learning_rate'].append(self.optim.param_groups[0]['lr'])
            stats['train_loss'].append(train_loss_tracker.avg)
            train_loss_tracker.reset()
            print("Begin Evaluation")
            # content_loss, perceptual_loss, loss, valid_psnr, valid_ssim = self.eval(valid_loader)
            loss, valid_psnr, valid_ssim = self.eval(valid_loader)

            stats['loss'].append(loss)
            stats['valid_psnr'].append(valid_psnr)
            stats['valid_ssim'].append(valid_ssim)
            best_psnr = max(stats['valid_psnr'])
            best_epoch = stats['valid_psnr'].index(best_psnr) + self.start_epoch

            epoch_time = int((datetime.now() - epoch_start).total_seconds())
            eva_time = int((datetime.now() - epoch_end).total_seconds())

            #####print
            print(
                'Epoch time : {} s| Evalu time : {} s| Valid Loss : {:.4f}|(Best: {:.2f} @epoch {})'.format(epoch_time,
                                                                                                            eva_time,
                                                                                                            loss,
                                                                                                            best_psnr,
                                                                                                            best_epoch))
            print('Valid PSNR : {:.2f} dB | Valid SSIM : {:.4f} | lr:{:.8f}'.format(valid_psnr,
                                                                                    valid_ssim,
                                                                                    self.optim.param_groups[0]['lr']))
            if _epoch % epoch_save == 0:
                if self.p.load_ckpt:
                    self.save_model(_epoch, best_psnr, best_epoch, stats)
                else:
                    self.save_model(_epoch, best_psnr, best_epoch, stats)

        train_time = str(datetime.now() - train_start)[:-7]
        print('Training done!  Total train time: {}\n'.format(train_time))

    def eval(self, valid_loader):
        '''Evaluate on validation set.'''
        self.model.train(False)

        # content_loss_tracker = Tracker()
        # perceptual_loss_tracker = Tracker()
        loss_tracker = Tracker()
        psnr_tracker = Tracker()
        ssim_tracker = Tracker()
        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.to(device)
                target = target.to(device)
                # source = source.type(torch.cuda.DoubleTensor)
                # target = target.type(torch.cuda.DoubleTensor)

            source_SR = self.model(source)

            content_loss = self.content_loss(source_SR, target)
            loss = content_loss
            loss_tracker.update(loss.item())

            # Compute PSRN, SSIM
            psnr_tracker.update(calc_psnr(source_SR, target, self.p.scale, 255))
            source_SR = np.transpose(source_SR.cpu().detach().squeeze(0).numpy(),[1,2,0])
            target = np.transpose(target.cpu().detach().squeeze(0).numpy(),[1,2,0])
            ssim_tracker.update(calculate_ssim(bgr2ycbcr(source_SR), bgr2ycbcr(target), self.p.scale))
        loss = loss_tracker.avg
        psnr_avg = psnr_tracker.avg
        ssim_avg = ssim_tracker.avg

        # return content_loss, perceptual_loss, loss, psnr_avg, ssim_avg
        return loss, psnr_avg, ssim_avg

    def test(self, test_loader):
        '''Test on test set.'''
        self.model.eval()

        print('Loading checkpoint from{}\n'.format(self.p.ckpt_file))
        # original saved file with DataParallel
        if self.use_cuda:
            state_dict = torch.load(self.p.ckpt_file)
            print('cuda')
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                if len(device_ids) > 1 and 'module.' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
            # load params
            self.model.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(self.p.ckpt_file, map_location=torch.device('cpu'))
            print('cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                # print(k, v)
                if 'module.' in k:
                    k = k[7:]  # remove `module.`
                    print('remove `module.`')
                new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)


        num_batches = len(test_loader)

        source_imgs = []
        result_imgs = []
        valid_loss_tracker = Tracker()
        psnr_tracker = Tracker()
        ssim_tracker = Tracker()

        for batch_idx, (source, target) in enumerate(test_loader):
            print(' : {}/{}'.format(batch_idx + 1, num_batches))

            source_imgs.append(source)
            print(source.shape)

            if self.use_cuda:
                source = source.to(device)
                target = target.to(device)
            source = source.type(torch.DoubleTensor)
            target = target.type(torch.DoubleTensor)

            print(source.shape)
            print(target.shape)

            source_SR = self.model(source)

            img_name = test_loader.dataset.imgs[batch_idx][:-4]
            img = source_SR.cpu().detach().squeeze(0).numpy()
            img = np.transpose(img, [1, 2, 0])
            # img = result_imgs[i]
            cv2.imwrite('{}/{}.png'.format('./../dataset/microscope_dataset/result', img_name), img)


            result_imgs.append(source_SR)

            psnr_tracker.update(calc_psnr(source_SR, target, self.p.scale, 255))
            # Compute PSRN, SSIM
            source_SR = np.transpose(source_SR.cpu().detach().squeeze(0).numpy(),[1,2,0])
            target = np.transpose(target.cpu().detach().squeeze(0).numpy(),[1,2,0])
            # psnr_tracker.update(compare_psnr(source_SR, target, 255))
            ssim_tracker.update(calculate_ssim(bgr2ycbcr(source_SR), bgr2ycbcr(target), self.p.scale))

        # valid_loss = valid_loss_tracker.avg
        psnr_avg = psnr_tracker.avg
        ssim_avg = ssim_tracker.avg

        # print('Valid Loss : {:.4f}'.format(valid_loss))
        print('Valid PSNR : {:.4f} dB'.format(psnr_avg))
        print('Valid SSIM : {:.4f}'.format(ssim_avg))


def calc_psnr(sr, hr, scale, rgb_range, test=True):
    # if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if test:
        shave = scale
        # print(diff.size)
        # print(diff.shape)
        if diff.shape[1] > 1:
                convert = diff.new(1, 3, 1, 1)
                convert[0, 0, 0, 0] = 65.738
                convert[0, 1, 0, 0] = 129.057
                convert[0, 2, 0, 0] = 25.064
                diff.mul_(convert).div_(256)
                diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, scale):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    # 确定相同的shape，shape输出维度的数值，几个数字代表每个维度的大小
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    else:
        # 减去shave值，去掉边界影响
        shave = scale
        img1 = img1[shave:-shave, shave:-shave]
        img2 = img2[shave:-shave, shave:-shave]
    # 确定输入的维度是几，维度为2则为Y通道，直接输出；维度为3，则确定第3个维度是不是无有有效数值，无效则去掉维度为1的部分
    '''
    numpy.squeeze(a,axis = None)
 1）a表示输入的数组；
 2）axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
 3）axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目；
 4）返回值：数组
 5) 不会修改原数组；'''
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()
    return rt


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
        dtype查看type
        astype转换type
    '''
    in_img_type = img.dtype
    img.astype(np.float64)
    if in_img_type != np.uint8:
        rlt = img * 255.0
    # convert
    if only_y:
        rlt = np.dot(rlt, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)

