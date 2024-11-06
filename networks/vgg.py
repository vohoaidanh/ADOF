import torch
import torch.nn.functional as F
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x

import torch.fft

class Conv2dFFT:
    def __init__(self, conv_layer):
        # Sử dụng trọng số từ lớp Conv2d bên ngoài
        self.weights = conv_layer.weight.clone().detach().to(torch.complex64)  # Chuyển trọng số sang kiểu phức
        self.kernel_size = self.weights.shape[2:]  # Kích thước kernel (chiều cao, chiều rộng)

    def forward(self, x):
        # Kích thước đầu vào
        batch_size, in_channels, height, width = x.shape
        
        # Tính kích thước padding để phép nhân trong miền tần số khớp với phép nhân chập thời gian
        padded_height = height + self.kernel_size[0] - 1
        padded_width = width + self.kernel_size[1] - 1

        # Biến đổi Fourier 3D của đầu vào với padding
        x_fft = torch.fft.fftn(x, s=(batch_size, padded_height, padded_width), dim=(-3, -2, -1))
        
        # Biến đổi Fourier 3D của trọng số (kernel) với padding
        weights_fft = torch.fft.fftn(self.weights, s=(batch_size, padded_height, padded_width), dim=(-3, -2, -1))
        
        # Nhân trong miền tần số
        output_fft = x_fft * weights_fft

        # Chuyển đổi ngược về miền thời gian
        output = torch.fft.ifftn(output_fft, dim=(-3, -2, -1)).real

        # Cắt output để khớp với kích thước mong muốn
        output = output[..., :height, :width]

        return output

if __name__ == "__main__":
    from torchsummary import summary
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from time import time
    model = VGG16()
    
    x = torch.rand(3,224,224)
    model(x.unsqueeze(0))
    summary(model,x.shape)
    
 
# =============================================================================
#     
#     
#     img = Image.open(r"C:\Users\danhv\Downloads\real.png")
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # Chuyển đổi thành tensor và scale từ [0, 255] sang [0.0, 1.0]
#     ])
#     #img = img.resize((255,255))
#     # Áp dụng phép biến đổi
#     image_tensor = transform(img)
#     
#     image_tensor = image_tensor[0:1]
# 
#     x = image_tensor
#     x.shape
#     conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2, bias=False)
#     x_conv1 = conv1(x)
#     x_conv1.shape
#     w = conv1.weight
#     w.shape
#     kernel = torch.fft.fftn(w, s=x.shape)
#   
#     kernel.shape
#     
#  
#     x_fft = torch.fft.fftn(x)
#     x_fft.shape
#     F_xw = x_fft * kernel[0]
#     
#     x_conv1_invert = torch.fft.ifftn(F_xw).real
#     x_conv1_invert.shape
# 
#     
#     plt.imshow(x_conv1_invert[0].detach())
#     print(x_conv1_invert[0].detach().sum())
#     plt.show()
#     plt.imshow(x_conv1.detach()[0]*4)
#     print(x_conv1.detach()[0].sum()*4)
#     plt.show()
#     
# 
# 
# =============================================================================


    #img = Image.open(r"C:\Users\danhv\Downloads\real.png")
    #transform = transforms.Compose([
    #    transforms.ToTensor(),  # Chuyển đổi thành tensor và scale từ [0, 255] sang [0.0, 1.0]
    #])
    #img = img.resize((255,255))
    # Áp dụng phép biến đổi
    #image_tensor = transform(img)
    #image_tensor = image_tensor.unsqueeze(0)
    #image_tensor = torch.rand(32,3,256,256)
    # Khởi tạo lớp Conv2d từ PyTorch
    #conv_layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3))
    # Đầu vào giả lập

    # Khởi tạo lớp Conv2dFFT với trọng số từ conv_layer
    #conv_fft = Conv2dFFT(conv_layer)
    
    # Tính toán
    #t = time()
    #output = conv_fft.forward(image_tensor)
    #print(time()-t)
    #output = output.permute(1,0,2,3)
    #print(output.shape)  # Kết quả sẽ có kích thước (1, 6, 32, 32)
    #t = time()
    #out2 = conv_layer(image_tensor)
    #print(time()-t)
    #print(out2.shape)  # Kết quả sẽ có kích thước (1, 6, 32, 32)

    
    #i=0
    #plt.imshow(output[0][i])
    #plt.imshow(out2.detach()[0][i])
    



