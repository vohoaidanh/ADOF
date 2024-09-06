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








import cv2
import torch
import urllib.request
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
    
    
filename = r"D:\dataset\biggan\0_real\999--n15075141_6442.png"

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)


with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

plt.imshow(output)



threshold = np.quantile(output, 0.60)
mask = output.copy()
#mask[mask >= threshold] = 0
mask = np.where(mask >= threshold, 0, 1)
plt.imshow(mask, cmap='gray')

img_tensor = img_tensor.unsqueeze(0)


with torch.no_grad():
    prediction = midas(img_tensor)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

plt.imshow(output)



threshold = np.quantile(output, 0.50)
mask = output.copy()
#mask[mask >= threshold] = 0
mask = np.where(mask >= threshold, 0, 1)
plt.imshow(mask, cmap='gray')

plt.imshow(img_tensor[0].permute(1,2,0))


