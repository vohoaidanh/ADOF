# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = r"C:\Users\danhv\Downloads\upscale\530--n03196217_3951.png"# Replace with your image path
image = Image.open(image_path)



img_arr = np.asarray(image)
x = img_arr[100,:,0]

fig, ax = plt.subplots(figsize=(20, 16))  # Set the desired figure size
# Set title and axis labels with increased font size
#ax.set_title('Resized Image', fontsize=30)   # Increase title font size
#ax.set_xlabel('X-Axis Label', fontsize=25)   # Increase x-axis label font size
#ax.set_ylabel('Y-Axis Label', fontsize=25)   # Increase y-axis label font size
ax.tick_params(axis='both', which='major', labelsize=25)  # Major ticks font size

# Plot the two graphs
ax.plot(x, label='Real', color='b')  # First graph

# Load the image
image_path = r"C:\Users\danhv\Downloads\upscale\Runway 2024-09-29T03_57_02.961Z Upscale Image Upscaled Image 1280 x 1280.jpg"  # Replace with your image path
image = Image.open(image_path)
image = image.resize((256,256))


img_arr = np.asarray(image)
x = img_arr[100,:,0]
ax.plot(x, label='Fake', color='r',linestyle='--',  alpha=0.9)  # Second graph

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




