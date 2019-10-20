'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models
import resnet
from types import MethodType

import warnings
warnings.filterwarnings("ignore")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def forward(self, x):
    x = self.conv1(x)
    skip4 = x
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    skip3 = x
    x = self.layer2(x)
    skip2 = x
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.upproj(x)
    x = self.upproj_cat1(x, skip2)
    x = self.upproj_cat2(x, skip3)
    x = self.upproj_cat3(x, skip4)

    return x

class UpProjection_layer(nn.Module):
    def __init__(self, in_channel, out_channel, height, width):
        super(UpProjection_layer, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1) #up sampling double width and height
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 =nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_skip = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn_skip = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.bn_skip(self.conv_skip(x))
        out = self.relu(skip+out)
        return out


class UpProjection_layer_cat(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpProjection_layer_cat, self).__init__()
        self.upsample = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv_cat = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_skip = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn_cat = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn_skip = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection):
        self.upsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, skip_connection), 1)
        out = self.relu(self.bn_cat(self.conv_cat(x)))
        out = self.bn2(self.conv2(out))
        skip = self.bn_skip(self.conv_skip(x))
        out = self.relu(skip+out)

        return out

class CSPN(nn.Module):
    def __init__(self):
        super(CSPN, self).__init__()

        #self.input_layer = nn.Linear(4 * 228 * 304, 3*224*224)

        self.resnet50 = models.resnet50(pretrained=True, progress=True)
        self.resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.dropout = nn.Dropout2d(p=0.9, inplace=True)   #dropout. To use it, should change forward()
        self.resnet50.upproj = UpProjection_layer(2048, 1024)
        self.resnet50.upproj_cat = UpProjection_layer_cat(1024, 512)
        self.resnet50.upproj_cat = UpProjection_layer_cat(512, 256)
        self.resnet50.upproj_cat = UpProjection_layer_cat(256, 64)
        self.resnet50.forward = MethodType(forward, self.resnet50)
        #self.resnet50.avgpool = Identity()                      #fine tuning
        #self.resnet50.fc = nn.Conv2d(2048, 1024, 1)    #layer link resnet50 and unsample layer

        self.blur_depth_layer = UpProjection_layer(64, 1)
        self.guid_depth_layer = UpProjection_layer(64, 8)
        self.Affinity_post_process_layer = Affinity_Propogation()

    def forward(self, x):
        out = self.resnet50(x)
        guid = self.guid_depth_layer(out)
        blur_depth = self.blur_depth_layer(out)

        out = self.Affinity_post_process_layer(guid, blur_depth, x[:,3,:,:])     #guidance, blured depth(predicted depth), sparse depth input
        return out


def test():
    net = CSPN()
    y = net(torch.randn(1, 4, 228, 304))
    print(y.size())

if __name__ == "__main__":
    test()
