import torch
import torch.quantization
from networks.resnet import resnet50

# Load and prepare the model
model = resnet50()
model.eval()
model.load_state_dict(torch.load('weights/ADOF_model_epoch_9.pth', map_location='cpu'), strict=True)


# Fuse layers for static quantization
model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']])
for layer_name, layer_module in model_fused.named_children():
    if "layer" in layer_name:
        for basic_block_name, basic_block_module in layer_module.named_children():
            torch.quantization.fuse_modules(basic_block_module, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']])

# Set the model's quantization configuration
model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # or 'qnnpack' for ARM

# Prepare the model for static quantization
torch.quantization.prepare(model_fused, inplace=True)

# Calibration: run a few batches of data through the model
sample_input = torch.randn(11, 3, 224, 224)  # Dummy input
with torch.no_grad():
    model_fused(sample_input)

# Convert the model to a quantized version
torch.quantization.convert(model_fused, inplace=True)





