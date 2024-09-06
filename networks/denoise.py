# -*- coding: utf-8 -*-
#https://github.com/cszn/KAIR/blob/master/models/network_dncnn.py

import torch.nn as nn
from collections import OrderedDict

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x-n
    
import segmentation_models_pytorch as smp

def UNet_denoise(activation=None):
    model = smp.Unet(
      encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
      encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
      in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
      classes=3,                      # model output channels (number of classes in your dataset)
      activation="softmax" # có thể thay đổi xem
    )
    return model

    
if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import matplotlib.pyplot as plt
    
    from PIL import Image
    import torchvision.transforms as transforms

    img_path = r"D:\dataset\biggan\0_real\235--n02106662_10101.png"
    img = Image.open(img_path)
    
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = t(img)


    #model = IRCNN(3,3,64)
    model = UNet_denoise()
    x = torch.rand(1,3,224,224)
    model.eval()
    y = model(img_tensor.unsqueeze(0))
    summary(model, input_size=img_tensor.size())
    plt.imshow(y[0][0].detach())
    yy = y[0].detach()
    plt.imshow((img_tensor ).permute(1,2,0))
    plt.imshow((img_tensor * yy).permute(1,2,0))
    loss = model.get













