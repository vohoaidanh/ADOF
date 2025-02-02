#!ls -l /root/.cache/huggingface/hub/

import torch
import torch.fft
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from functools import partial
import timm


__all__ = ['build_model']
import torch
import torch.fft

def highpass_filter(image_tensor: torch.Tensor, cutoff_percent: float) -> torch.Tensor:
    """
    Preprocess a tensor image by converting it to the frequency domain, applying a high-frequency cutoff,
    and converting it back to the spatial domain.

    Parameters:
    - image_tensor (torch.Tensor): Input image tensor of shape (bz, c, 224, 224).
    - cutoff_percent (float): Percentage of high-frequency components to keep (0 to 100).

    Returns:
    - torch.Tensor: The processed image tensor in the spatial domain.
    """
    # Ensure the input is in the correct shape (bz, c, 224, 224)
    assert image_tensor.ndim == 4, "Image tensor should be 4D (bz, c, 224, 224)."
    
    bz, c, h, w = image_tensor.shape
    
    # Perform 2D FFT on each image channel
    fft_image = torch.fft.fft2(image_tensor, dim=(-2, -1))
    fft_image_shifted = torch.fft.fftshift(fft_image, dim=(-2, -1))  # Shift zero freq component to the center
    
    # Create a mask to remove low frequencies and keep high frequencies
    cutoff = cutoff_percent / 100.0
    center_h, center_w = h // 2, w // 2
    cutoff_h, cutoff_w = int((1 - cutoff) * h // 2), int((1 - cutoff) * w // 2)

    # Start with a mask that keeps everything (high frequencies included)
    mask = torch.ones_like(fft_image, dtype=torch.bool)
    
    # Remove the low frequencies in the center based on cutoff
    mask[:, :, center_h-cutoff_h:center_h+cutoff_h, center_w-cutoff_w:center_w+cutoff_w] = False
    
    # Apply the mask: Keep only high frequencies
    fft_image_filtered = fft_image_shifted * mask
    
    # Shift back and perform inverse FFT to return to spatial domain
    fft_image_unshifted = torch.fft.ifftshift(fft_image_filtered, dim=(-2, -1))
    image_filtered = torch.fft.ifft2(fft_image_unshifted, dim=(-2, -1)).real
    
    return image_filtered

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


def ADOF_diff(input_tensor):
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
    
    #diff = diff_y/diff_x
    
    # Set the gradient values to 0.0 for cases where dy/dx is greater than 100, which approximates gradients close to +/- π/2
    # The threshold of 100 is a reference value and can be tuned for optimal performance during actual training.
    #diff = torch.where(torch.abs(diff) > 1e2, torch.tensor(0.0), diff)
    
    # Compute the arctangent of the difference and normalize it to the range [0, 1]
    #output = (torch.arctan(diff) / (torch.pi / 2) + 1.0) / 2.0
    output = (torch.arctan2(diff_y,diff_x) / (torch.pi / 2) + 1.0) / 2.0
    return torch.abs(output[:, 0] - output[:, 2]).unsqueeze(1)

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

class Detector(nn.Module):
    def __init__(self, backbone, num_features = 'auto', num_classes=1, pretrained=False, freeze_exclude=None):
        super(Detector, self).__init__()
        self.preprocess = lambda x: x
 
        if isinstance(backbone, str):
            if backbone.lower() == 'adof':
                self.backbone = resnet50(pretrained=False)
                self.backbone.adof = lambda x: x  # Định nghĩa hàm không làm gì
                self.preprocess = ADOF
                in_features = self.backbone(torch.randn(1, 3, 224, 224))

            else:
                self.backbone = resnet50(pretrained=False)
                self.backbone.adof = lambda x: x  # Định nghĩa hàm không làm gì
                self.preprocess = lambda x: x
                in_features = self.backbone(torch.randn(1, 3, 224, 224))

        elif isinstance(backbone, nn.Module):
          # Sử dụng mô hình đã cho trực tiếp
            self.backbone = backbone
        else:
            raise TypeError("backbone_name must be a string or a nn.Module instance")
               
        
        in_features = in_features.shape[1]

        if isinstance(freeze_exclude, list):
            self.freeze_layers(self.backbone, freeze_exclude)
        
        if num_features != 'auto':
            in_features = int(num_features)
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.preprocess(x)
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_layers(self, model, layers_to_freeze):
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True

def build_model(**kwargs):
    model = Detector(**kwargs)
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
    backbone = 'adof_diff'
    #backbone = resnet50(pretrained=False)
    model = build_model(backbone=backbone, pretrained=False, num_classes=1, freeze_exclude=None)
    model.preprocess(torch.rand(2,3,224,224)).shape
    print(model(torch.rand(2,3,224,224)))
    
# =============================================================================
#     
#     img = Image.open(r"C:\Users\danhv\Downloads\ffhq_real.png")
#     img = t(img)
#     img = img.unsqueeze(0)
#     
#     out = highpass_filter(img, cutoff_percent=50)[0].permute(1,2,0)
#     plt.imshow(100*out)
#     
#     out_100 = highpass_filter(img, cutoff_percent=10)[0].permute(1,2,0)
#     out_75 = highpass_filter(img, cutoff_percent=75)[0].permute(1,2,0)
#     out_30 = highpass_filter(img, cutoff_percent=30)[0].permute(1,2,0)
#     
#     torch.sum(img[0].permute(1,2,0) - out_30)
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    







