import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayerBlock(nn.Module):
    def __init__(self, input):
        super(DenseLayerBlock, self).__init__()
        self.f1 = nn.ReLU(inplace = True)
        self.f2 = nn.BatchNorm2d(input)
        self.f3 = nn.Conv2d(input, 32, 3, stride=1, padding=1)
        self.f4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.f5 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.f6 = nn.Conv2d(96, 32, 3, stride=1, padding=1)
        self.f7 = nn.Conv2d(128, 32, 3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.f1(self.f3(self.f2(x)))
        conv2 = self.f1(self.f4(conv1))
        conv2_dense = self.f1(torch.cat([conv1, conv2], 1))

        conv3 = self.f1(self.f5(conv2_dense))
        conv3_dense = self.f1(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.f1(self.f6(conv3_dense))
        conv4_dense = self.f1(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.f1(self.f7(conv4_dense))
        conv5_dense = self.f1(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return conv5_dense

class TransitionLayerBlock(nn.Module):
    def __init__(self, input, output):
        super(TransitionLayerBlock, self).__init__()

        self.f1 = nn.ReLU(inplace = True)
        self.f2 = nn.BatchNorm2d(output)
        self.f3 = nn.Conv2d(input, output, 1)
        self.f4 = nn.AvgPool2d(2, stride=2, padding=0)

    def forward(self, x):
        return self.f4(self.f2(self.f1(self.f3(x))))

class DenseNet(nn.Module):

    def __init__(self, w, h):
        super(DenseNet, self).__init__()

        self.w = w
        self.h = h

        self.f1 = nn.Conv2d(1, 64, 7, padding=3)
        self.f2 = nn.ReLU()

        layers = []
        layers.append(DenseLayerBlock(64))
        self.f3 = nn.Sequential(*layers)

        layers = []
        layers.append(DenseLayerBlock(128))
        self.f4 = nn.Sequential(*layers)

        layers = []
        layers.append(DenseLayerBlock(128))
        self.f5 = nn.Sequential(*layers)

        layers = []
        layers.append(TransitionLayerBlock(160, 128))
        self.f6 = nn.Sequential(*layers)

        layers = []
        layers.append(TransitionLayerBlock(160, 128))
        self.f7 = nn.Sequential(*layers)

        layers = []
        layers.append(TransitionLayerBlock(160, 64))
        self.f8 = nn.Sequential(*layers)

        """self.f9 = nn.BatchNorm2d(64)
        self.f10 = nn.Linear(64*4*4, 512)
        self.f11 = nn.Linear(512, classes)"""

    def forward(self, x):
        out = self.f2(self.f1(x))

        out = self.f3(out)
        out = self.f6(out)

        out = self.f4(out)
        out = self.f7(out)

        out = self.f5(out)
        out = self.f8(out)

        out = out.view(1, 1, self.w, self.h)
        """out = self.f9(out)
        out = out.view(-1, 64*4*4)
        out = self.f11(self.f10(out))"""

        return out
