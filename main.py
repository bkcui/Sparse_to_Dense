from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader

import os
import argparse

from model import Sparse_to_dense_net
import nyu_dataloader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

#from models import *
#from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch nyudataset Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--n_samples', default=200, type=int, help='Sparse sample size')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--checkpoint', default='checkpoint/ResNet50.t7', type=str, help='Checkpoint location')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--predict', action='store_true', help='forward prop')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = None  # best test accuracy
loss_list = []  #for saving loss value
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
#print('==> Preparing data..')

nyudepth_data_train = nyu_dataloader.NyuDepthDataset(csv_file='data/nyudepth_hdf5_2/nyudepth_hdf5_train.csv',
                                   root_dir='.',
                                   split='train',
                                   n_sample=args.n_samples,
                                   input_format='hdf5')
nyudepth_data_val = nyu_dataloader.NyuDepthDataset(csv_file='data/nyudepth_hdf5_2/nyudepth_hdf5_val.csv',
                                   root_dir='.',
                                   split='val',
                                   n_sample=args.n_samples,
                                   input_format='hdf5')

nyudataset_train = dataloader.DataLoader(nyudepth_data_train, batch_size=args.batch_size, shuffle=True, sampler=None,
                                         batch_sampler=None, num_workers=0)
nyudataset_val = dataloader.DataLoader(nyudepth_data_val, batch_size=args.batch_size, shuffle=False, sampler=None,
                                       batch_sampler=None, num_workers=0)
#nyudataset_test = dataloader.DataLoader(nyudepth_data_val, batch_size=1, shuffle=False, sampler=None,
#                                       batch_sampler=None, num_workers=0)



# Model
#print('==> Building model..')

# net = models.VGG('VGG19')
# net = models.ResNet50()
# net = models.PreActResNet18()
# net = models.GoogLeNet()
# net = models.DenseNet121()
# net = models.ResNeXt29_2x64d()
# net = models.MobileNet()
# net = models.MobileNetV2()
# net = models.DPN92()
# net = models.ShuffleNetG2()
# net = models.SENet18()
# net = models.ShuffleNetV2(1)
# net = models.shake_net()
net = Sparse_to_dense_net()
net = net.to(device)
#checkpoint = torch.load('checkpoint/bidense3.t7')
#net.load_state_dict(checkpoint['net'])
#best_acc = checkpoint['acc']
#start_epoch = checkpoint['epoch']


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    print('loss :', best_loss, 'epoch :', start_epoch)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, data in enumerate(nyudataset_train):
        input_data = data['rgbd'].to(device)
        target = data['depth'].to(device)
        optimizer.zero_grad()
        outputs = net(input_data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print('Loss : ', loss.item())

#validate
def test(epoch):
    global best_loss
    net.eval()
    test_loss = 0
    global loss_list

    with torch.no_grad():
        for batch_idx, data in enumerate(nyudataset_val):
            inputs = data['rgbd'].to(device)
            targets = data['depth'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

    loss_list.append(test_loss)
    if best_loss == None:       #for epoch 0
        best_loss = test_loss + 1

    # Save checkpoint.
    print('Val Loss : ', test_loss)
    if best_loss > test_loss:
        with open("Loss.txt", 'a') as Loss_list:
            for item in loss_list:
                Loss_list.write("%s\n" % item)
        print('Saving..  %f' % test_loss)
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        torch.save(state, args.checkpoint)
        best_loss = test_loss
        loss_list = []

'''
def predict():
    net.eval()

    fig = plt.figure()
    with torch.no_grad():
        for batch_idx, input_data in enumerate(nyudataset_test):
            input_data_rgbd = input_data['rgbd'].to(device)
            outputs = net(input_data_rgbd)
            depth = transforms.ToPILImage()(outputs.cpu().view(1, 228, 304))

            ax = plt.subplot(3, 1,1)
            ax.axis('off')
            plt.imshow(transforms.ToPILImage()(input_data['rgb_ori'][0, :, :, :].view(3, 228, 304)))

            ax = plt.subplot(3, 1, 2)
            ax.axis('off')
            plt.imshow(transforms.ToPILImage()(input_data['depth'].view(1, 228, 304)))

            ax = plt.subplot(3, 1, 3)
            ax.axis('off')
            plt.imshow(depth)

            plt.show()
            input()
'''



if __name__ == '__main__':


    learning_rate = args.lr

    if args.predict:    #on test set
        #predict()
        pass

    else:
        for epoch in range(start_epoch, start_epoch+20):

            if learning_rate > 0.001 and epoch%10 == 0: #learning rate dacay
                learning_rate -= 0.001
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            train(epoch)
            test(epoch)