from __future__ import absolute_import, division, print_function

import math
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

#-------------------------------------------------------------------------#
#   主干: 卷积+bn+relu -> 卷积+bn+relu -> 卷积+bn
#   短接: 卷积+bn
#   短接后有relu
#-------------------------------------------------------------------------#
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # change
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        # relu是共用的
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#-----------------------------------------------------------------#
#   使用Renset50作为主干特征提取网络，最终会获得一个
#   16x16x2048的有效特征层
#-----------------------------------------------------------------#
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 512,512,3 -> 256,256,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # 256,256,64 -> 128,128,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change

        # 128,128,64 -> 128,128,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 128,128,256 -> 64,64,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 64,64,512 -> 32,32,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # 32,32,1024 -> 16,16,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)      # nn.AdaptiveAvgPool2d((1, 1)) 这样更好,7是专门为224设计的 224/32=7
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   步长不为1或者进出通道不相等就设置下采样层 conv+bn 没有relu
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            # conv+bn
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        # 第一次有短接层
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 后面没有短接层
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 512,512,3   -> 256,256,64
        x = self.maxpool(x) # 256,256,64  -> 128,128,64

        x = self.layer1(x)  # 128,128,64  -> 128,128,256
        x = self.layer2(x)  # 128,128,256 -> 64,64,512
        x = self.layer3(x)  # 64,64,512   -> 32,32,1024
        x = self.layer4(x)  # 32,32,1024  -> 16,16,2048

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained = True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir = 'model_data/')
        model.load_state_dict(state_dict)
    #----------------------------------------------------------#
    #   获取特征提取部分
    #----------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features = nn.Sequential(*features)
    # print(features)
    return features


#----------------------------------------------------------#
#   加强特征提取部分,使用转置卷积
#   每次转置卷积后特征层的宽高变为原来的两倍。
#   $HW_{out} = (HW_{in} - 1) * stride - 2 * padding + kernel\_size$
#   32 = (a-1) * 2 - 2 * 1 + 4 = 2x - 2 - 2 + 4 = 2a
#----------------------------------------------------------#
class resnet50_Decoder(nn.Module):
    def __init__(self, in_channels, bn_momentum=0.1):
        super(resnet50_Decoder, self).__init__()
        self.in_channels = in_channels
        self.bn_momentum = bn_momentum
        self.deconv_with_bias = False

        #----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        #----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers =3,              # 重复3次
            num_filters=[256, 128, 64], # 3次的out_channels
            num_kernels=[4, 4, 4],      # 3次的kernel_size
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel_size  = num_kernels[i] # kernel_size
            out_channels = num_filters[i] # out_channels

            layers.append(
                nn.ConvTranspose2d(
                    in_channels =self.in_channels,
                    out_channels=out_channels,
                    kernel_size =kernel_size,
                    stride      =2,
                    padding     =1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(out_channels, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)    # 128,128,64


#----------------------------------------------------------#
#   头部分
#----------------------------------------------------------#
class resnet50_Head(nn.Module):
    def __init__(self, num_classes=80, hidden_channels=64, bn_momentum=0.1):
        super(resnet50_Head, self).__init__()
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes   分类
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, 2             框的宽高回归
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, 2             中心位置回归
        #-----------------------------------------------------------------#

        #-----------------------------------------------------------------#
        # 热力图预测部分    3x3Conv+BN+Relu + 1x1Conv
        #-----------------------------------------------------------------#
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1))

        #-----------------------------------------------------------------#
        # 宽高预测的部分    3x3Conv+BN+Relu + 1x1Conv
        #-----------------------------------------------------------------#
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1))

        #-----------------------------------------------------------------#
        # 中心点预测的部分  3x3Conv+BN+Relu + 1x1Conv
        #-----------------------------------------------------------------#
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()    # 分类数都变为0~1之间
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        # 分类,宽高,中心
        return hm, wh, offset


if __name__ == "__main__":
    resnet50(pretrained=False)