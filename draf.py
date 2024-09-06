# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_path = r"D:\dataset\biggan\1_fake\00199642.png"
img = Image.open(img_path)

img_tensor = t(img)


# Instantiate the loss function
loss_fn = nn.BCEWithLogitsLoss()

# Example logits (raw outputs from the model)
pre = [1 for i in range(32)]
pre = torch.tensor(pre, dtype=torch.float32)
logits = torch.tensor(pre, dtype=torch.float32)  # Size should match batch_size

# Example labels
labels = [1 for i in range(32)]
labels = torch.tensor(labels, dtype=torch.float32)  # Size should match batch_size

# Compute the loss
loss = loss_fn(logits, labels)
print(loss)

torch.sigmoid(pre)

import segmentation_models_pytorch as smp


# Khởi tạo mô hình
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,  # RGB
    activation=None
)

out = model(img_tensor.unsqueeze(0))

plt.imshow(img_tensor.permute(1,2,0))

plt.imshow(out[0].detach().permute(1,2,0))














