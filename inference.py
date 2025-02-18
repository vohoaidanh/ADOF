# -*- coding: utf-8 -*-

import os
import torch
import numpy as np   
import pandas as pd
from torchvision import transforms
from PIL import Image
from networks.resnet import resnet50
from tqdm import tqdm


# Load model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load("weights/ADOF_model_epoch_9.pth", map_location="cpu"), strict=True)
#model.cuda()
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về đúng kích thước cho ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

# Folder chứa ảnh và file CSV output
image_folder = r"D:\\thay_nghia_dataset\\earthquakes\\earthquakes\\close-up view"
csv_output = "earthquakes.csv"

# Lưu kết quả
results = []
real_count = 0 
generate_count = 0 
# Lặp qua từng ảnh trong thư mục
for img_name in tqdm(os.listdir(image_folder)[-10:]):
    img_path = os.path.join(image_folder, img_name)
    
    # Kiểm tra nếu file là ảnh
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    
    # Load và preprocess ảnh
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    
    # Dự đoán
    with torch.no_grad():
        logits = model(image).cpu().flatten().numpy()  # Giá trị trước sigmoid
        #output = model(image).cpu().numpy().flatten()[0]  # Chuyển output về dạng scalar
        output = model(image).cpu().sigmoid().flatten()[0].numpy()
        

    
    # Lưu kết quả
    label = ['real', 'generate']
    y_hat =  (output>=0.5)*1
    if (y_hat==1):
        generate_count+=1
    else:
        real_count+=1
        
    results.append([img_name,y_hat, label[y_hat]] )
print(f"Real count:{real_count}, Generate count: {generate_count}")
# Lưu logits vào file Numpy (.npy)
logits_file_path = 'logits.npy'
np.save(logits_file_path, logits)

# Ghi kết quả vào CSV
pd.DataFrame(results, columns=["Image", "Prediction","Label"]).to_csv(csv_output, index=False)

print(f"Inference hoàn tất! Kết quả được lưu vào {csv_output}")










