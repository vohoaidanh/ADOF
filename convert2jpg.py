# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.nn import functional as F
from io import BytesIO
import os
from tqdm import tqdm  # Thêm thư viện tqdm


def png2jpeg(image, quality=100):
    # Chuyển sang RGB nếu ảnh có kênh alpha
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    
    # Tạo một đối tượng BytesIO để lưu ảnh
    jpg_image = BytesIO()
    
    image.save(jpg_image, format="JPEG", quality=quality)
    jpg_image.seek(0)
    image = Image.open(jpg_image)
    
    return image

def convert_dataset(src_folder, dst_folder):
    # Lấy tất cả các file trong thư mục nguồn
    all_files = []
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                all_files.append(os.path.join(root, file))

    # Sử dụng tqdm để theo dõi tiến trình
    for src_path in tqdm(all_files, desc="Converting images", unit="file"):
        # Lấy tên thư mục tương đối
        relative_path = os.path.relpath(os.path.dirname(src_path), src_folder)
        dst_dir = os.path.join(dst_folder, relative_path)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Đổi tên file thành .jpg và lưu ảnh chuyển đổi
        dst_path = os.path.join(dst_dir, os.path.basename(src_path).rsplit(".", 1)[0] + ".jpg")
        
        with Image.open(src_path) as img:
            for i in range(100):
                img = png2jpeg(img)
            img.save(dst_path, "JPEG", quality=100)
            
import argparse            
def main():
    # Dùng argparse để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Convert images from PNG to JPEG")
    parser.add_argument('src', type=str, help="Path to the source folder")
    parser.add_argument('dst', type=str, help="Path to the destination folder")
    args = parser.parse_args()
    
    # Chạy hàm với tham số từ dòng lệnh
    convert_dataset(args.src, args.dst)

if __name__ == '__main__':
    main()
    #python script_name.py "D:\datasets\datasets" "D:\datasets\biggan_jpg"














