#!ls -l /root/.cache/huggingface/hub/

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from functools import partial
import timm
from random import choice


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

def mean_filter_2d(input_tensor, kernel_size=5):
    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.size()

    # Define the mean filter kernel for 3 channels
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
    
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)

    output = F.conv2d(input_tensor, kernel, padding=1, groups=channels)
    
    return output


class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()
        self.backbone = backbone
        _, self.num_features, _, _ = self.backbone(torch.rand(1,3,224,224)).shape

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(-1, self.num_features)
        return features

from networks.resnet import resnet50

class ExtractFeature(nn.Module):
    def __init__(self, backbone, pretrained=True, get_feature=False, num_classes=1, freeze_exclude=None, num_features=None):
        super(ExtractFeature, self).__init__()
        self.reference = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.reference = nn.Sequential(*list(self.reference.children())[:-1])
        
        self.investigate = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.investigate = nn.Sequential(*list(self.investigate.children())[:-1])
        
        self.get_feature = get_feature
        self.preprocess = ADOF
        self.prelu = nn.PReLU()
        self.num_classes = num_classes
        # Một lớp fully connected kết hợp hai vector (sau khi concatenate)
        self.fc1 = nn.Linear(512, 256)  # Kết hợp 2 vector (512 + 512)
        self.classifier = nn.Linear(256, self.num_classes)
            
    def forward(self, x):
        x1, x2 = x
        x1 = self.preprocess(x1)
        x2 = self.preprocess(x2)
               
        x1 = self.reference(x1)
        x2 = self.investigate(x2)

        # Kết hợp query và key (concatenate)
        #x = torch.cat([x1, x2], dim=-1)  # Đầu ra (batch_size, 512 * 2)
        x = x1 + x2  # Cộng trực tiếp (batch_size, 512)
        # Đưa qua các lớp fully connected để học sự tương đồng
        x = self.prelu(self.fc1(x))
        output = self.classifier(x)
        
        return output
       

# =============================================================================
#     def _split_horizontal(self,x):
#         # img_tensor: [batch_size, channels, 224, 224]
#         bz, c, h, w = x.shape
#     
#         # Chia theo chiều ngang, mỗi ảnh sẽ có kích thước [batch_size, channels, 112, 224]
#         top_half = x[:, :, :h//2, :]   # Nửa trên
#         bottom_half = x[:, :, h//2:, :]  # Nửa dưới
#         
#         return top_half, bottom_half
#     
# =============================================================================

class Detector(nn.Module):
    def __init__(self, backbone, num_features = 'auto', num_classes=1, pretrained=False, freeze_exclude=None):
        super(Detector, self).__init__()
        self.preprocess = lambda x: x
        # Tạo backbone từ timm
        self.c = 3
        if isinstance(backbone, str):
            if backbone.lower() == 'adof':
                self.backbone = resnet50(pretrained=False)
                self.backbone.adof = lambda x: x  # Định nghĩa hàm không làm gì
                self.preprocess  = ADOF
            else:
                self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        
        elif isinstance(backbone, nn.Module):
          # Sử dụng mô hình đã cho trực tiếp
            self.backbone = backbone
        else:
            raise TypeError("backbone_name must be a string or a nn.Module instance")
        
        
        in_features = self.backbone(torch.randn(1, self.c, 224, 224))
        in_features = in_features.shape[1]

        if isinstance(freeze_exclude, list):
            self.freeze_layers(self.backbone, freeze_exclude)
        
        if num_features != 'auto':
            in_features = int(num_features)
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x1 = self.preprocess(x)
        x1 = self.sppf(x1)
        if self.c == 6:
            x2 = self.adofcross(x)
            x1 = torch.cat((x1, x2), dim=1)
            
        features = self.backbone(x1)
        output = self.classifier(features)
        
        return output
    
    def freeze_layers(self, model, layers_to_freeze):
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True


def build_model(**kwargs):
    #model = Detector(**kwargs)
    model = ExtractFeature(**kwargs)
    return model

if __name__  == '__main__':
    from torchsummary import summary
    from torchvision.transforms import transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    t = transforms.ToTensor()
    all_model_list = timm.list_models()

    vit_list = timm.list_models(filter='vit*')
    vgg_list = timm.list_models(filter='vgg*')
    eff_list = timm.list_models(filter='ef*')
    mobilenet = timm.list_models(filter='*mobilenet*')
    RegNet = timm.list_models(filter='*RegNet*')
    resnet_list = timm.list_models(filter='resn*')

    #'vgg19_bn', 'vit_base_patch32_224', 'efficientnet_b0', 'efficientvit_b0', 'mobilenetv3_large_100', 'mobilenetv3_small_100', 'mobilenetv3_small_050'
    
    backbone = 'scattering'
    #backbone = resnet50(pretrained=False)
    model = build_model(backbone=backbone, pretrained=False, num_classes=1, freeze_exclude=None)

    model.adof(torch.rand(2,3,224,224)).shape
    print(model(torch.rand(2,3,224,224)))

    backbone = timm.create_model('resnet18', pretrained=False)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    
    input_tensor = torch.randn(1, 3, 224, 224)
    features = backbone(input_tensor)
    plt.plot(features.detach().ravel())
    
    extract_feature = build_model(backbone = 'resnet18', pretrained=False)
    extract_feature.eval()
    features = extract_feature((input_tensor, input_tensor))



