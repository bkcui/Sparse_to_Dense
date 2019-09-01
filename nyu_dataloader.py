# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
import h5py

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
imagenet_eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
imagenet_eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]], dtype=np.float32)


class NyuDepthDataset(Dataset):
    # nyu depth dataset
    def __init__(self, csv_file, root_dir, split, n_sample=200, input_format='img'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images. (super dir of data folder)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rgbd_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.input_format = input_format
        self.n_sample = n_sample

    def __len__(self):
        return len(self.rgbd_frame)

    def __getitem__(self, idx):
        # read input image (img format and hdf5 format)
        if self.input_format == 'img':
            #            print('==> Input Format is image')
            rgb_name = os.path.join(self.root_dir,
                                    self.rgbd_frame.iloc[idx, 0])
            with open(rgb_name, 'rb') as fRgb:
                rgb_image = Image.open(rgb_name).convert('RGB')

            depth_name = os.path.join(self.root_dir,
                                      self.rgbd_frame.iloc[idx, 1])
            with open(depth_name, 'rb') as fDepth:
                depth_image = Image.open(depth_name)

        # read input hdf5
        elif self.input_format == 'hdf5':
            #            print('==> Input Format is hdf5')
            file_name = os.path.join(self.root_dir,
                                     self.rgbd_frame.iloc[idx, 0])
            rgb_h5, depth_h5 = self.load_h5(file_name)
            #            print(depth_h5.dtype)
            rgb_image = Image.fromarray(rgb_h5, mode='RGB')
            depth_image = Image.fromarray(depth_h5.astype('float32'), mode='F')
        #            plt.figure()
        #            show_img(rgb_image)
        #            plt.figure()
        #            show_img(depth_image)
        else:
            print('error: the input format is not supported now!')
            return None

        _s = np.random.uniform(1.0, 1.5)
        s = np.int(240 * _s)
        degree = np.random.uniform(-5.0, 5.0)

        #data arguemntation
        if self.split == 'train':
            tRgb = transforms.Compose([transforms.Resize(s),
                                           transforms.RandomRotation((degree, degree)),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           #                                           data_transform.Lighting(0.1, imagenet_eigval, imagenet_eigvec)])
                                           transforms.CenterCrop((228, 304)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])

            tDepth = transforms.Compose([transforms.Resize(s),
                                             transforms.RandomRotation((degree, degree)),
                                             transforms.CenterCrop((228, 304))])
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            if np.random.uniform() < 0.5:
                rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
                depth_image = depth_image.transpose(Image.FLIP_LEFT_RIGHT)

            rgb_image = transforms.ToTensor()(rgb_image)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
            else:
                depth_image = transforms.ToTensor()(depth_image)
            depth_image = depth_image.div(_s)
            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)


        elif self.split == 'val':
            tRgb = transforms.Compose([transforms.Resize(240),
                                           transforms.CenterCrop((228, 304)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])
            tRgb_ori = transforms.Compose([transforms.Resize(240),
                                           transforms.CenterCrop((228, 304)),
                                           transforms.ToTensor(),
                                           transforms.ToPILImage()])

            tDepth = transforms.Compose([transforms.Resize(240),
                                             transforms.CenterCrop((228, 304))])
            rgb_image_ori = tRgb_ori(rgb_image)
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            rgb_image = transforms.ToTensor()(rgb_image)
            rgb_image_ori = transforms.ToTensor()(rgb_image_ori)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
            else:
                depth_image = transforms.ToTensor()(depth_image)
            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)
            sample = {'rgbd': rgbd_image, 'depth': depth_image, 'rgb_ori' : rgb_image_ori}
            return sample

        sample = {'rgbd': rgbd_image, 'depth': depth_image}
        return sample

    def createSparseDepthImage(self, depth_image, n_sample):
        sparse_mask = torch.ones(1, depth_image.size(1), depth_image.size(2))    #tensor shape (channel, height, width)
        probaility = n_sample/(depth_image.size(1)*depth_image.size(2))             #probability = sample size / num_of pixels
        sparse_mask = sparse_mask*probaility
        #print(sparse_mask)
        sparse_mask = torch.bernoulli(sparse_mask)
        sparse_depth = torch.mul(depth_image, sparse_mask)
        #print(depth_image.size(1)*depth_image.size(2))
        return sparse_depth

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        #    print (f.keys())
        rgb = f['rgb'][:].transpose(1, 2, 0)
        depth = f['depth'][:]
        return (rgb, depth)


def show_img(image):
    """Show image"""
    plt.imshow(image)


def test_imgread():
    # train preprocessing
    nyudepth_dataset = NyuDepthDataset(csv_file='data/nyudepth_hdf5_2/nyudepth_hdf5_train.csv',
                                           root_dir='.',
                                           split = 'train',
                                           n_sample = 200,
                                           input_format='hdf5')
    #nyudepth_dataset = NyuDepthDataset(csv_file='data/nyudepth_hdf5/nyudepth_hdf5_val.csv',
    #                                   root_dir='.',
    #                                   split='val',
    #                                   n_sample=500,
    #                                   input_format='hdf5')
    fig = plt.figure()
    for i in range(4):                 #len(nyudepth_dataset)         #read four image
        sample = nyudepth_dataset[i]
        rgb = transforms.ToPILImage()(sample['rgbd'][0:3, :, :])
        depth = transforms.ToPILImage()(sample['depth'])
        sparse_depth = transforms.ToPILImage()(sample['rgbd'][3, :, :].unsqueeze(0))
        depth_mask = transforms.ToPILImage()(torch.sign(sample['depth']))
        sparse_depth_mask = transforms.ToPILImage()(sample['rgbd'][3, :, :].unsqueeze(0).sign())
        print(sample['rgbd'][0:3, :, :])
        invalid_depth = torch.sum(sample['rgbd'][3, :, :].unsqueeze(0).sign() < 0)
        print(invalid_depth)
        #        print(sample['depth'].size())
        #        print(torch.sign(sample['sparse_depth']))
        ax = plt.subplot(5, 4, i + 1)
        ax.axis('off')
        show_img(rgb)
        ax = plt.subplot(5, 4, i + 5)
        ax.axis('off')
        show_img(depth)
        ax = plt.subplot(5, 4, i + 9)
        ax.axis('off')
        show_img(depth_mask)
        ax = plt.subplot(5, 4, i + 13)
        ax.axis('off')
        show_img(sparse_depth)

        ax = plt.subplot(5, 4, i + 17)
        ax.axis('off')
        show_img(sparse_depth_mask)
        plt.imsave('sparse_depth.png', sparse_depth_mask)
        if i == 3:
            plt.show()
            break


if __name__ == "__main__":
    nyudepth_dataset = NyuDepthDataset(csv_file='data/nyudepth_hdf5_2/nyudepth_hdf5_train.csv',
                                       root_dir='.',
                                       split='train',
                                       n_sample=200,
                                       input_format='hdf5')
    data = DataLoader(nyudepth_dataset, batch_size=16, shuffle=True, sampler=None,
                          batch_sampler=None, num_workers=2)        #check batch

    for i, dat in enumerate(data):

        print(dat['rgbd'].size())
        fig = plt.figure()
        rgb = dat['rgbd'][:, 0:3, :, :]
        print(rgb.size())
        rgb = transforms.ToPILImage()(rgb[0, :, :, :].view(3, 228, 304))    #check image
        show_img(rgb)
        plt.show()
        input()
    #test_imgread()