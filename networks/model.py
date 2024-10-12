#!ls -l /root/.cache/huggingface/hub/

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np


__all__ = ['build_model']

def ADOF(input_tensor):
    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.size()

    # Define the gradient filters for x and y directions
    kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_x.transpose(2, 3)  # Transpose kernel_x to get kernel_y

    # Expand the kernels to match the number of input channels
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)

    # Apply the filters
    diff_x = F.conv2d(input_tensor, kernel_x, padding=1, groups=channels) + 1e-9 # to avoid div 0
    diff_y = F.conv2d(input_tensor, kernel_y, padding=1, groups=channels)
    
    diff = diff_y/diff_x
    
    # Set the gradient values to 0.0 for cases where dy/dx is greater than 100, which approximates gradients close to +/- π/2
    # The threshold of 100 is a reference value and can be tuned for optimal performance during actual training.
    diff = torch.where(torch.abs(diff) > 1e2, torch.tensor(0.0), diff)
    
    # Compute the arctangent of the difference and normalize it to the range [0, 1]
    output = (torch.arctan(diff) / (torch.pi / 2) + 1.0) / 2.0
  
    return output

def ADOFCross(input_tensor):
    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.size()

    # Define the gradient filters for x and y directions
    kernel_x = torch.tensor([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Expand the kernels to match the number of input channels
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)

    # Apply the filters
    diff_x = F.conv2d(input_tensor, kernel_x, padding=1, groups=channels) + 1e-9 # to avoid div 0
    diff_y = F.conv2d(input_tensor, kernel_y, padding=1, groups=channels)
    
    diff = diff_y/diff_x
    
    # Set the gradient values to 0.0 for cases where dy/dx is greater than 100, which approximates gradients close to +/- π/2
    # The threshold of 100 is a reference value and can be tuned for optimal performance during actual training.
    #diff = torch.where(torch.abs(diff) > 1e2, torch.tensor(0.0), diff)
    
    # Compute the arctangent of the difference and normalize it to the range [0, 1]
    output = (torch.arctan(diff) / (torch.pi / 2) + 1.0) / 2.0
  
    return output

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 4]):
        super(SPPF, self).__init__()
        self.pool_sizes = pool_sizes
        
        # Define convolutional layer after pooling
        self.conv = nn.Conv2d(in_channels * (len(pool_sizes) + 1), out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        # List to hold the pooled outputs
        pools = []
        
        # Perform pooling for each size and append to the list
        for size in self.pool_sizes:
            pooled = nn.functional.avg_pool2d(x, kernel_size=size, stride=size)
            # Upsample the pooled output to match the input size
            pooled = nn.functional.interpolate(pooled, size=x.shape[2:], mode='nearest')
            pools.append(pooled)
        
        # Concatenate the original input with pooled outputs
        out = torch.cat([x] + pools, dim=1)
        out = self.conv(out)
        out = self.bn1(out)
        return out


class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()
        self.backbone = backbone
        _, self.num_features, _, _ = self.backbone(torch.rand(1,3,224,224)).shape

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(-1, self.num_features)
        return features

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,backbone=None, num_features=None, freeze_exclude=None):
        super(ResNet, self).__init__()
        
        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.adof = ADOF
        self.attn = SelfAttention(in_channels=256)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _zoom(self,input_tensor, scale=0.75):
        if scale==1:
            return input_tensor
        batch_size, channels, height, width = input_tensor.shape
        
        # Tính toán kích thước của crop (75% của height và width)
        crop_height = int(scale * height)
        crop_width = int(scale * width)
        
        # Tính toán tọa độ crop giữa ảnh
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        bottom = top + crop_height
        right = left + crop_width
        
        # Thực hiện center crop
        cropped_tensor = input_tensor[:, :, top:bottom, left:right]

        resized_tensor = F.interpolate(cropped_tensor, size=(height, width), mode='bilinear', align_corners=False)
        
        return resized_tensor
        
 
    def forward(self, x):
        x = [self._zoom(x, scale) for scale in [1, 0.75, 0.5]]
        xs = []
        for x_i in x:
            x_i = self.adof(x_i)
            x_i = self.conv1(x_i)
            x_i = self.bn1(x_i)
            x_i = self.relu(x_i)
            x_i = self.maxpool(x_i)
            x_i = self.layer1(x_i)
            xs.append(x_i)
        
        x = self.attn(*xs)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, k, q):
        batch_size, C, width, height = x.size()
        # Tính Q, K, V
        query = self.query_conv(q).view(batch_size, -1, width * height).permute(0, 2, 1)
        #print('query:', query.shape)
        key = self.key_conv(k).view(batch_size, -1, width * height)
        #print('key:', key.shape)

        value = self.value_conv(x).view(batch_size, -1, width * height)
        #print('value:', value.shape)

        # Tính attention map
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # Tính output
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Áp dụng trọng số gamma
        out = self.gamma * out + x
        return out

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def build_model(**kwargs):
    model = resnet50(**kwargs)
    return model

if __name__  == '__main__':
    from torchsummary import summary
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    t = transforms.ToTensor()
    #all_model_list = timm.list_models()

    #vit_list = timm.list_models(filter='vit*')
    #vgg_list = timm.list_models(filter='vgg*')
    #eff_list = timm.list_models(filter='ef*')
    #mobilenet = timm.list_models(filter='*mobilenet*')
    #RegNet = timm.list_models(filter='*RegNet*')
    #resnet_list = timm.list_models(filter='resn*')

    #'vgg19_bn', 'vit_base_patch32_224', 'efficientnet_b0', 'efficientvit_b0', 'mobilenetv3_large_100', 'mobilenetv3_small_100', 'mobilenetv3_small_050'
    
    #backbone = 'cnndetection'
    #backbone = resnet50(pretrained=False)
    model = build_model(pretrained=False, num_classes=1)
    
    image_path = r"C:\Users\danhv\Downloads\upscale\Runway 2024-09-29T03_57_02.961Z Upscale Image Upscaled Image 1280 x 1280.jpg"  # Replace with your image path
    image = Image.open(image_path)
    image = image.resize((256,256))
    image = t(image)
    model(image.unsqueeze(0))
    
    summary(model, input_size=(3,224,224))

    
    