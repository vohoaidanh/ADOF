from torchvision.models import resnet18
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # Tính Q, K, V
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        print('query:', query.shape)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        print('key:', key.shape)

        value = self.value_conv(x).view(batch_size, -1, width * height)
        print('value:', value.shape)

        # Tính attention map
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # Tính output
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Áp dụng trọng số gamma
        out = self.gamma * out + x
        return out

class ResNet18WithAttention(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18WithAttention, self).__init__()
        # Tải mô hình ResNet18 đã có sẵn
        self.resnet = resnet18(pretrained=True)

        # Thêm self-attention sau layer 2
        self.attention = SelfAttention(in_channels=128)  # 128 là số kênh đầu ra của layer 2

        # Điều chỉnh lớp fully connected cuối cùng cho bài toán phân loại
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Các lớp ban đầu của ResNet18
        x = self.resnet.conv1(x)
        print(x.shape)
        x = self.resnet.bn1(x)
        print(x.shape)

        x = self.resnet.relu(x)
        print(x.shape)

        x = self.resnet.maxpool(x)
        print(x.shape)

        # Layer 1
        x = self.resnet.layer1(x)
        print(x.shape)

        # Layer 2 + Attention
        x = self.resnet.layer2(x)
        print(x.shape)

        x = self.attention(x)
        print(x.shape)

        # Các layer còn lại
        x = self.resnet.layer3(x)
        print(x.shape)

        x = self.resnet.layer4(x)
        print(x.shape)

        # Average Pooling và lớp fully connected cuối cùng
        x = self.resnet.avgpool(x)
        print(x.shape)

        x = torch.flatten(x, 1)
        print(x.shape)

        x = self.resnet.fc(x)
        print(x.shape)

        return x


modele = ResNet18WithAttention(num_classes=1)

modele(torch.rand(1,3,224,224))







