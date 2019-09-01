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

import warnings
warnings.filterwarnings("ignore")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Sparse_to_dense_net(nn.Module):
    def __init__(self):
        super(Sparse_to_dense_net, self).__init__()

        #self.input_layer = nn.Linear(4 * 228 * 304, 3*224*224)

        self.resnet50 = resnet.resnet50()
        self.resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.resnet50.avgpool = Identity()                      #fine tuning
        self.resnet50.fc = nn.Conv2d(2048, 1024, 1)    #layer link resnet50 and unsample layer

        self.unsample_layer = nn.Sequential(nn.ConvTranspose2d(1024, 512, 3, stride=2,padding=1, output_padding=1),
                                             nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
                                             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                             nn.Conv2d(64, 1, 3, padding=1)
                                             )

    def forward(self, x):
        out = self.resnet50(x)
        out = self.unsample_layer(out)
        out = F.interpolate(out, size=(228, 304), mode='bilinear')
        return out


def test():
    net = Sparse_to_dense_net()
    y = net(torch.randn(1, 4, 228, 304))
    print(y.size())

if __name__ == "__main__":
    test()
