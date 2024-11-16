import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.nn import functional as F
from io import BytesIO

t = transforms.Compose([
    transforms.ToTensor()
    ])

def png2jpeg(image):
    # Chuyển sang RGB nếu ảnh có kênh alpha
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    
    # Tạo một đối tượng BytesIO để lưu ảnh
    jpg_image = BytesIO()
    
    image.save(jpg_image, format="JPEG", quality=100)
    jpg_image.seek(0)
    image = Image.open(jpg_image)
    
    return image


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
    return torch.abs(output[:, 0] - output[:, 2])


def diff(intensor):
    return torch.abs(intensor[:, 0] - intensor[:, 2])


real_path = r"D:\datasets\real_gen_dataset\train\0_real\000609815.jpg"
fake_path = r"C:\Users\danhv\Downloads\467021852_3890142961221882_8487593993681782116_n.jpg"

img_real = Image.open(fake_path)
img_real=img_real.crop((0,0,256,256))
img_real_jpg = png2jpeg(img_real)
img_real_tensor = t(img_real).unsqueeze(0)

adof_real = ADOF(img_real_tensor)
diff_real = adof_real
plt.imshow(diff_real[0][0:,0:])
diff_real.sum()
diff_real.mean()

plt.imshow(img_real_tensor[0][2][300:400,300:400])


img_real_jpg = png2jpeg(img_real)
tensor2 = torch.tensor(0.0)
for i in range(100):
    img_real_jpg = png2jpeg(img_real_jpg)
    if i%10 == 0:
        img_real_tensor = t(img_real_jpg).unsqueeze(0)
        are_close = torch.all(torch.abs(img_real_tensor - tensor2) <  1e-2)
        adof_real = ADOF(img_real_tensor)
        diff_real = diff(adof_real)
        plt.imshow(diff_real[0:,0:])
        plt.show()
        #plt.imshow(torch.nn.functional.normalize((img_real_tensor-tensor2)[0].permute(1,2,0)))
        #plt.show()
        tensor2=img_real_tensor
        print(are_close)
        


img_real_tensor = t(img_real_jpg).unsqueeze(0)

adof_real = ADOF(img_real_tensor)
diff_real = diff(adof_real)
plt.imshow(adof_real[0].permute(1,2,0))
diff_real.sum()

diff_real.mean()








