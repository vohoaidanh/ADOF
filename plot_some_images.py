# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def mean_filter_2d(input_tensor, kernel_size=5):
    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.size()

    # Define the mean filter kernel for 3 channels
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
    
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    pad = (kernel_size-1)//2
    output = F.conv2d(input_tensor, kernel, padding=pad, groups=channels)
    
    return output

# Ví dụ sử dụng
# image = torch.randn(1, 1, 256, 256)  # Một ảnh ngẫu nhiên kích thước 256x256

t = transforms.ToTensor()
# Load the image
image_path = r"D:\K32\do_an_tot_nghiep\THESIS\Material\ffhq_real.png"# Replace with your image path
image = Image.open(image_path)
img_arr = np.asarray(image)
a= t(image).unsqueeze(0)
a = mean_filter_2d(a, kernel_size=7)
x = a[0][0][128,:]

fig, ax = plt.subplots(figsize=(20, 16))  # Set the desired figure size
# Set title and axis labels with increased font size
#ax.set_title('Resized Image', fontsize=30)   # Increase title font size
#ax.set_xlabel('X-Axis Label', fontsize=25)   # Increase x-axis label font size
#ax.set_ylabel('Y-Axis Label', fontsize=25)   # Increase y-axis label font size
ax.tick_params(axis='both', which='major', labelsize=25)  # Major ticks font size

# Plot the two graphs
ax.plot(x, label='Real', color='b')  # First graph

# Load the image
image_path = r"D:\K32\do_an_tot_nghiep\THESIS\Material\ffhq_fake_4_small.png"  # Replace with your image path
image = Image.open(image_path)
img_arr = np.asarray(image)
a= t(image).unsqueeze(0)
a = mean_filter_2d(a, kernel_size=7)
x = a[0][0][128,:]

ax.plot(x, label='Fake', color='r',linestyle='-',  alpha=0.9)  # Second graph

ax.legend(fontsize=25)  # Display the legend with specified font size


###########################
# Caculate some correlation
###########################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main(image_path = r"D:\Downloads\dataset\CNN_synth\biggan\1_fake\00219321.png", size=None):
# Load the image
    # Replace with your image path
    image = Image.open(image_path)
    r = 100
    if size is not None:
        image = image.resize(size=size)
    #plt.imshow(image)
    #plt.show()
    
    img_arr = np.asarray(image)
    
    
    plt.figure(figsize=(30,20))
    rows = img_arr[:,r,0:3]
    # Plot the red, green, and blue channels separately
    plt.plot(rows[:, 0], color='r', label='Red channel')
    #plt.plot(rows[:, 1], color='g', label='Green channel')
    #plt.plot(rows[:, 2], color='b', label='Blue channel')
    #plt.legend(fontsize=20)
    
    print('image size is:', img_arr.shape)
    return rows[:, 0]


a = main(r"D:\K32\do_an_tot_nghiep\THESIS\Material\ffhq_real.png")
b = main(r"D:\K32\do_an_tot_nghiep\THESIS\Material\ffhq_fake_4.png", (256,256))

plt.figure(figsize=(30,20))
plt.plot(a[:],label='(1)', color='red')
plt.plot(b[:],label='(2)', color='green')
plt.legend(fontsize=40, loc='upper left')
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)



image = Image.open(r"D:\K32\do_an_tot_nghiep\THESIS\Material\ffhq_fake_4.png").resize(size=(256,256))
image.save(r"D:\K32\do_an_tot_nghiep\THESIS\Material\ffhq_fake_4_small.png")
    
######################
image = Image.open(r"D:\K32\do_an_tot_nghiep\THESIS\Material\ffhq_fake_4.png").resize(size=(224,224))

import torch
from torchvision import transforms

# Bước 1: Định nghĩa các phép biến đổi
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Thay đổi kích thước hình ảnh
    transforms.ToTensor(),            # Chuyển đổi hình ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

image_tensor_origin = transform(image)
image_tensor = image_tensor_origin.permute(1, 2, 0)  # Chuyển đổi từ [C, H, W] sang [H, W, C]
#image_tensor = image_tensor.clamp(0, 1)


plt.imshow(np.array(image))
plt.imshow(image_tensor.numpy())  # Chuyển đổi tensor sang numpy array
    
from networks.resnet import ADOF
from networks.model import ADOFCross,ADOF

adof = ADOF(image_tensor_origin.unsqueeze(0))

plt.imshow(adof.squeeze(0).permute(1,2,0))  # Chuyển đổi tensor sang numpy array

plt.plot(np.array(image)[100,:,0])
plt.plot(image_tensor[100,:,0])
plt.plot(adof[0].permute(1,2,0)[100,:,0])

torch.var(image_tensor[100,:,0])

torch.var(adof[0].permute(1,2,0)[100,:,0])

torch.std(adof[0].permute(1,2,0)[100,:,0])

red_chanel = adof[0][0]

plt.imshow(red_chanel)
plt.plot(red_chanel[100,:])
mean_per_row = torch.mean(red_chanel, axis=1)
plt.plot(mean_per_row[100])

torch.mean(red_chanel[100,:])

variance = torch.var(red_chanel, axis=1)
plt.plot(variance)

device='cpu'
from torch.nn import functional as F
def Mean(input_tensor):
    device = input_tensor.device
    batch_size, channels, height, width = input_tensor.size()

    # Define the gradient filters for x and y directions
    kernel_x = torch.ones((3,3), dtype=torch.float32, device=device)

    # Expand the kernels to match the number of input channels
    kernel_x = kernel_x.expand(channels, 1, 3, 3)

    # Apply the filters
    output = F.conv2d(input_tensor, kernel_x, padding=1, groups=channels) 
    
    return output
    
#(1,3,224,224) , (3,1,3,3)
mean_tensor = Mean(image_tensor_origin.unsqueeze(0))
plt.imshow(mean_tensor.squeeze(0).permute(1,2,0))  # Chuyển đổi tensor sang numpy array

plt.figure(figsize=(20, 10))
plt.plot(mean_tensor[0][0][100,:]/9)
plt.plot(image_tensor_origin[0][100,:])

plt.plot(image_tensor_origin[0][100,:] - mean_tensor[0][0][100,:]/9)
a = image_tensor_origin[0] - mean_tensor[0][0]/9
plt.imshow(a)


plt.plot(a[100,:])


kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(0)
kernel_y = kernel_x.transpose(2, 3)  # Transpose kernel_x to get kernel_y

# Expand the kernels to match the number of input channels
kernel_x = kernel_x.expand(3, 1, 3, 3)
kernel_y = kernel_y.expand(3, 1, 3, 3)
##################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to compute the power spectrum for a given image
def compute_power_spectrum(image_array):
    fft_image = np.fft.fft2(image_array)
    fft_shift = np.fft.fftshift(fft_image)  # Shift the zero frequency component to the center
    power_spectrum = np.abs(fft_shift)**1
    return power_spectrum

# Path to the folder containing the images
folder_path = r'D:\Downloads\dataset\CNN_synth\gaugan\1_fake'  # Replace with your folder path

# List to store power spectra of all images
power_spectra_list = []

# Loop over all images in the folder
for filename in os.listdir(folder_path)[:300]:
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Check for image file formats
        image_path = os.path.join(folder_path, filename)
        
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.crop((0, 0, 224, 224))  # Crop to a square based on the smallest dimension
        image_array = np.array(image)
        
        # Compute power spectrum for this image
        power_spectrum = compute_power_spectrum(image_array)
        power_spectra_list.append(power_spectrum)

# Compute the mean power spectrum across all images
mean_power_spectrum = np.mean(np.array(power_spectra_list), axis=0)
#normalized_mean_power_spectrum = (mean_power_spectrum - np.min(mean_power_spectrum)) / (np.max(mean_power_spectrum) - np.min(mean_power_spectrum))

# Áp dụng log với một hệ số điều chỉnh để giảm độ sáng của tần số thấp
log_power_spectrum = np.log1p(mean_power_spectrum)

# Cắt và giới hạn giá trị (clipping) để tránh tần số thấp quá sáng
clipped_log_power_spectrum = np.clip(log_power_spectrum, a_min=None, a_max=np.percentile(log_power_spectrum, 99))  # Giới hạn tại 99th percentile

# Chuẩn hóa giá trị để tần số cao nổi bật hơn
#normalized_spectrum = (clipped_log_power_spectrum - np.min(clipped_log_power_spectrum)) / (np.max(clipped_log_power_spectrum) - np.min(clipped_log_power_spectrum))

# =============================================================================
# # Hiển thị phổ công suất
# plt.figure(figsize=(20,20))
# plt.imshow(normalized_spectrum)
# plt.title(folder_path)
# plt.axis('off')
# plt.show()
# 
# =============================================================================

# Chuyển đổi ma trận thành một dãy các giá trị đơn lẻ
flattened_spectrum = log_power_spectrum.flatten()

# Vẽ biểu đồ histogram
plt.figure(figsize=(10,10))
plt.hist(flattened_spectrum, bins=200, color='blue', alpha=0.7)
plt.title(folder_path)
plt.xlabel("Log Power Spectrum")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



