import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPool3x3(nn.Module):
    def __init__(self, in_planes, planes, stride, dynamic_pool=False, pool_first=False):
        super(DynamicPool3x3, self).__init__()
        self.pool_first = pool_first
        self.dynamic_pool = dynamic_pool

        if self.dynamic_pool:
            self.conv = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            pool_planes = in_planes if self.pool_first else planes
            self.dynamic_pool = nn.Conv2d(pool_planes, pool_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=pool_planes)
        else:
            self.conv = conv3x3(in_planes, planes, stride)
    
    def forward(self, x):
        if not self.dynamic_pool:
            x = self.conv(x)
        elif self.pool_first:
            x = self.dynamic_pool(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.dynamic_pool(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dynamic_pool=False, pool_first=False):
        super(BasicBlock, self).__init__()
        self.conv1 = DynamicPool3x3(inplanes, planes, stride, dynamic_pool, pool_first)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DynamicPool3x3(planes, planes, 1, dynamic_pool, pool_first)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, ns, num_classes, pool_first=False, dynamic_pool=False):
        super(ResNetCifar, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model

        self.layers = []
        self.is_imageNet = False
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        for p, n in zip((16, 32, 64), ns):
            self.layers += [*self._make_layer(block, p, n, stride=1 if p == 16 else 2, pool_first=pool_first, dynamic_pool=dynamic_pool)]
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.feat_length = 64 * block.expansion

        self.layers = nn.Sequential(*self.layers)


    def _make_layer(self, block, planes, blocks, stride=1, pool_first=False, dynamic_pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pool_first=pool_first, dynamic_pool=dynamic_pool))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, pool_first=pool_first, dynamic_pool=dynamic_pool))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        # x = self.layer1(x)  # 32x32
        # x = self.layer2(x)  # 16x16
        # x = self.layer3(x)  # 8 x 8
        
        x = self.layers(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet_cifar110(pool_first=False, dynamic_pool=False):
    block = BasicBlock
    ns = (110 - 2) // 6
    ns = [ns] * 3
    num_classes = 10
    return ResNetCifar(block=block, ns=ns, num_classes=num_classes, pool_first=pool_first, dynamic_pool=dynamic_pool)