import torch

# Giả sử bạn có mô hình với các lớp Conv, BatchNorm và ReLU
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    torch.nn.BatchNorm2d(16),
    torch.nn.ReLU()
)
print(model)
model = model.eval()
# Hợp nhất các lớp Conv, BN và ReLU thành một lớp duy nhất
model2 = torch.quantization.fuse_modules(model, [['0', '1', '2']])

# Kiểm tra mô hình sau khi hợp nhất
print(model2)
x=torch.rand(1,3,5,5)
model(x)[0][0] - model2(x)[0][0]

from fvcore.nn import FlopCountAnalysis
FlopCountAnalysis(model, x).total()
FlopCountAnalysis(model2, x).total()

import torch.quantization as tq
from networks.model import build_model

# List các lớp cần hợp nhất (theo thứ tự Conv -> BatchNorm -> ReLU)
modules_to_fuse = [
    ['conv1', 'bn1', 'relu'],  # Hợp nhất Conv2d, BatchNorm2d và ReLU ở layer đầu tiên
    ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'],  # Layer 1.0: Bottleneck đầu tiên
    ['layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu'],
    ['layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu'],
    ['layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu'],
    # Thêm các lớp khác nếu cần
]


model = build_model(backbone='color', pretrained=False, num_classes=1, freeze_exclude=None)

model2 = torch.quantization.fuse_modules(model, [['0', '1', '2']])












