import torch
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms.functional as tvF

import os
import glob
# from PIL import Image
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import random
from torch.utils.data.distributed import DistributedSampler

def load_train_dataset(args, single=False):
    root_dir = args.train_data_dir
    train = args.train
    transform = args.transform
    dataset = Dataset(root_dir=root_dir, train=train, valid=False, test=False, transform=transform, ext='_LR_x4.png',
                      lr_dir='x4BI', hr_dir='x4', digit=-10, patch_size=48)

    if single:
        return DataLoader(dataset, batch_size=1, sampler=DistributedSampler(dataset))
        # return DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def load_valid_dataset(args, single=True):
    root_dir = args.valid_data_dir
    transform = args.transform
    dataset = Dataset(root_dir=root_dir, valid=True, train=False, test=False, transform=False, ext='_LR_x4.png',
                      lr_dir='x4BI', hr_dir='x4', digit=-10)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def load_test_dataset(args):
    root_dir = args.test_data_dir
    train = args.train
    transform = args.transform
    dataset = Dataset(root_dir=root_dir, train=False, valid=False, test=True, transform=transform, ext='_LR_x4.png',
                      lr_dir='x4BI', hr_dir='x4', digit=-10)
    return DataLoader(dataset, batch_size=1, shuffle=False)


class Dataset(Dataset):  # 子类化

    def __init__(self, root_dir, train, valid, test, transform, ext, lr_dir, hr_dir, digit, scale=4, patch_size=48):
        """初始化变量"""
        self.imgs = os.listdir(os.path.join(root_dir, hr_dir))  # 用于测试图片时的命名

        self.train = train
        self.valid = valid
        self.set_path = make_dataset(root_dir, train, ext, lr_dir, hr_dir, digit)
        self.transform = transform
        self.test = test
        self.scale = scale
        self.patch = patch_size

    def __getitem__(self, idx):
        """加载图像，整数索引"""
        if self.transform:
            source, target = self.set_path[idx]
            source_img = np.expand_dims(load_img(source), axis=2)
            target_img = np.expand_dims(load_img(target), axis=2)
            source_img, target_img = image_test_crop(source_img, target_img, scale=self.scale)
            source_img, target_img = np2Tensor([source_img, target_img], 255)
            return source_img, target_img

        if self.train:
            source, target = self.set_path[idx]
            source_img = cv2.imread(source)
            target_img = cv2.imread(target)
            source_img, target_img = get_patch(source_img, target_img, patch_size=self.patch, scale=self.scale)
            source_img, target_img = augment([source_img, target_img])
            source_img, target_img = np2Tensor([source_img, target_img], 255)
            return source_img, target_img

        if self.valid:
            source, target = self.set_path[idx]
            source_img = cv2.imread(source)
            target_img = cv2.imread(target)
            source_img, target_img = image_test_crop(source_img, target_img, scale=self.scale)
            source_img, target_img = np2Tensor([source_img, target_img], 255)

            return source_img, target_img
        if self.test:
            source, target = self.set_path[idx]

            source_img = cv2.imread(source)
            target_img = cv2.imread(target)
            source_img, target_img = image_test_crop(source_img, target_img, scale=self.scale)
            source_img, target_img = np2Tensor([source_img, target_img], 255)

            return source_img, target_img


    def __len__(self):
        """返回数据集大小"""

        return len(self.set_path)



def make_dataset(root_dir, train, ext, lr_dir, hr_dir, digit):
    dataset = []

    # if train:
    dir_source = os.path.join(root_dir, lr_dir)
    dir_target = os.path.join(root_dir, hr_dir)

    for img in glob.glob(os.path.join(dir_target, '*.png')):
        target_img_name = os.path.basename(img)
        source_img_name = target_img_name[:digit] + ext
        dataset.append([os.path.join(dir_source, source_img_name), img])

    # else:
    #     for img in glob.glob(os.path.join(root_dir, '*.png')):
    #         dataset.append(img)

    return dataset


def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw = img_in.shape[:2]

    p = scale
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def image_valid_crop(source_img, target_img, scale, piexl=128):
    h, w, _ = source_img.shape
    source_img = source_img[0:piexl, 0:piexl, ::]
    # source = source_img[0:h*scale, 0:w*scale, :]
    target = target_img[0:piexl * scale, 0:piexl * scale, ::]
    # source = cv2.resize(target, (128, 128), interpolation=cv2.INTER_CUBIC)

    return source_img, target


def image_test_crop(source_img, target_img, scale):
    h, w, _ = source_img.shape
    # source = source_img[0:h*scale, 0:w*scale, :]
    target = target_img[0:h * scale, 0:w * scale, ::]
    # source = cv2.resize(target, (128, 128), interpolation=cv2.INTER_CUBIC)

    return source_img, target


def image_resize(source_img, target_img):
    source = cv2.resize(source_img, (48, 48), interpolation=cv2.INTER_CUBIC)
    target = cv2.resize(target_img, (48 * 4, 48 * 4), interpolation=cv2.INTER_CUBIC)
    return source, target


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.0)

        return tensor

    return [_np2Tensor(_l) for _l in l]
