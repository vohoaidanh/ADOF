import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_gradient(image):
    """
    Tính gradient theo hướng x và y, và độ lớn của gradient cho một ảnh.
    """
    # Tính gradient theo hướng x và y bằng sai phân hữu hạn
    gradient_x = np.diff(image, axis=1)  # Gradient theo trục x
    gradient_y = np.diff(image, axis=0)  # Gradient theo trục y

    # Để hình ảnh gradient cùng kích thước với ảnh gốc, cần thêm một cột và một hàng giá trị 0
    gradient_x = np.pad(gradient_x, ((0, 0), (0, 1)), 'constant')
    gradient_y = np.pad(gradient_y, ((0, 1), (0, 0)), 'constant')

    # Tính độ lớn của gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orient = np.arctan2(gradient_y, gradient_x)

    return gradient_x, gradient_y, magnitude, orient

def process_folder(folder_path):
    """
    Tính gradient cho tất cả các ảnh trong một thư mục và trả về các giá trị độ lớn gradient.
    """
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Đọc ảnh dưới dạng grayscale
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                # Tính gradient
                _, _, magnitude, orient = compute_gradient(image)
                #
                results.append(orient)

    return results

def plot_gradient_distribution(magnitudes,title='', ylim=1e9):
    """
    Vẽ biểu đồ phân bố của độ lớn gradient cho tất cả các ảnh.
    """
    # Gộp tất cả các độ lớn gradient thành một mảng duy nhất
    all_magnitudes = np.concatenate([m.ravel() for m in magnitudes])

    # Vẽ biểu đồ phân bố
    plt.figure(figsize=(10, 6))
    plt.hist(all_magnitudes, bins=365, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Frequency')
    plt.ylim(0, ylim)
    plt.show()
    return all_magnitudes

# Đường dẫn đến thư mục chứa ảnh
folder_path = r'D:\Downloads\dataset\CNN_synth\stylegan\car\1_fake'

# Tính gradient cho các ảnh trong thư mục
results = process_folder(folder_path)

# Vẽ phân bố độ lớn gradient
m = plot_gradient_distribution(results,folder_path, 2e5)













