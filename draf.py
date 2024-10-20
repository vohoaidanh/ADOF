# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

# Dữ liệu mẫu
x = np.arange(-10,10,1)
y = x.copy()
y[y<0] = 0




# Tính đa thức Lagrange
polynomial = lagrange(x, y)

# Vẽ đồ thị
x_vals = np.linspace(-10, 10, 20)
y_vals = polynomial(x_vals)

plt.plot(x, y, 'ro', label='Dữ liệu')
plt.plot(x_vals, y_vals, label='Đa thức nội suy')
plt.legend()
plt.show()

plt.plot( x, 'b.', label='Dữ liệu')

plt.plot( y, 'r-', label='Dữ liệu')

x = np.arange(-10,10,0.1)
y = np.arange(-10,10,0.1)
y[y<0] = np.sin(y[y<0])
plt.plot(x,y, 'r-', label='Dữ liệu')

N = len(y)
T = 1.0  # Thời gian lấy mẫu (có thể điều chỉnh)
Y = np.fft.fft(y)
freq = np.fft.fftfreq(N, T)



# Vẽ đồ thị
plt.plot( np.abs(Y))
plt.title("Biến Đổi Fourier")
plt.xlabel("Tần số")
plt.ylabel("Biên độ")
plt.show()




import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread(r"C:\Users\danhv\Downloads\real.png", 0)  # Đọc ảnh xám

# Kích thước kernel là một hàm sin
rows, cols = image.shape
x = np.linspace(0, 2 * np.pi, cols)
y = np.linspace(0, 2 * np.pi, rows)

# Tạo kernel sin 2D bằng cách nhân sin(x) và sin(y)
sin_kernel_x = np.sin(x)
sin_kernel_y = np.sin(y)
sin_kernel_2d = np.outer(sin_kernel_y, sin_kernel_x)  # Tạo kernel 2D

# Áp dụng tích chập
filtered_image = cv2.filter2D(image, -1, sin_kernel_2d)

# Hiển thị ảnh gốc và ảnh sau tích chập
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Ảnh sau khi tích chập')

plt.show()

plt.plot(sin_kernel_y.reshape(-1))

sin_kernel_2d.max()




import numpy as np

# Define two 3x3 matrices representing the noisy image (x) and the denoised image (y)
x = np.array([[10, 12, 14],
              [9,  11, 15],
              [8,  10, 16]])

y = np.array([[9,  11, 13],
              [10, 12, 14],
              [7,  11, 15]])

# Calculate the 2D L2 norm (E) between x and y
E = 0.5 * np.sum((x - y) ** 2)
E


import re
import json

# Input string mimicking the content of a file
input_text = """
Saving model ./checkpoints/adof-carReal-carFake-resnet18-2024_10_17_08_46_38/model_epoch_15.pth
 Epoch 15 : 2024_10_17_09_02_16 --> 2024_10_17_09_03_11
Saving model ./checkpoints/adof-carReal-carFake-resnet18-2024_10_17_08_46_38/model_epoch_last.pth
(Val @ epoch 15) acc: 1.0; ap: 1.0
*************************
2024_10_17_09_03_12
(0 car         ) acc: 99.8; ap: 100.0; r_acc: 1.0; f_acc: 1.0
(1 cat         ) acc: 99.2; ap: 100.0; r_acc: 1.0; f_acc: 1.0
(2 chair       ) acc: 99.5; ap: 100.0; r_acc: 1.0; f_acc: 1.0
(3 horse       ) acc: 98.0; ap: 99.7; r_acc: 1.0; f_acc: 1.0
(4 Mean      ) acc: 99.1; ap: 99.9
*************************
2024_10_17_09_03_18
2024_10_17_09_03_36 Train loss: 0.002707220846787095 at step: 2300 lr 0.00014580000000000002
2024_10_17_09_03_55 Train loss: 0.0005876213544979692 at step: 2350 lr 0.00014580000000000002
Saving model ./checkpoints/adof-carReal-carFake-resnet18-2024_10_17_08_46_38/model_epoch_16.pth
 Epoch 16 : 2024_10_17_09_03_18 --> 2024_10_17_09_04_13
Saving model ./checkpoints/adof-carReal-carFake-resnet18-2024_10_17_08_46_38/model_epoch_last.pth
(Val @ epoch 16) acc: 1.0; ap: 1.0
*************************
2024_10_17_09_04_14
(0 car         ) acc: 99.8; ap: 100.0; r_acc: 1.0; f_acc: 1.0
(1 cat         ) acc: 98.8; ap: 100.0; r_acc: 1.0; f_acc: 1.0
(2 chair       ) acc: 99.2; ap: 100.0; r_acc: 1.0; f_acc: 1.0
(3 horse       ) acc: 97.5; ap: 99.8; r_acc: 1.0; f_acc: 1.0
(4 Mean      ) acc: 98.8; ap: 99.9
*************************
2024_10_17_09_04_20
2024_10_17_09_04_22 Train loss: 0.007870160043239594 at step: 2400 lr 0.00014580000000000002
2024_10_17_09_04_41 Train loss: 0.00630034226924181 at step: 2450 lr 0.00014580000000000002
2024_10_17_09_05_00 Train loss: 0.00909484177827835 at step: 2500 lr 0.00014580000000000002
Saving model ./checkpoints/adof-carReal-carFake-resnet18-2024_10_17_08_46_38/model_epoch_17.pth
 Epoch 17 : 2024_10_17_09_04_20 --> 2024_10_17_09_05_15
Saving model ./checkpoints/adof-carReal-carFake-resnet18-2024_10_17_08_46_38/model_epoch_last.pth
(Val @ epoch 17) acc: 1.0; ap: 1.0
"""

# Define regex pattern to extract all the data columns
pattern = r"\((\d+)\s+(\w+)\s+\)\s+acc:\s([\d.]+);\s+ap:\s([\d.]+);\s+r_acc:\s([\d.]+);\s+f_acc:\s([\d.]+)"

file_path = r"D:\K32\do_an_tot_nghiep\THESIS\Material\origin-resnet18-carReal-horseFake.txt"  # Thay thế bằng đường dẫn tới file của bạn
with open(file_path, 'r') as file:
    input_text = file.read()


# Find all matches using regex
matches = re.findall(pattern, input_text)

# Create a list of dictionaries
data_list = []
for match in matches:
    data_dict = {
        #"index": int(match[0]),
        "name": match[1],
        "acc": float(match[2]),
        "ap": float(match[3]),
        "r_acc": float(match[4]),
        "f_acc": float(match[5])
    }
    data_list.append(data_dict)

# Convert the list of dictionaries to JSON
json_data = json.dumps(data_list, indent=4)
parsed_data = json.loads(json_data)

# Print the resulting JSON
print(json_data)



import matplotlib.pyplot as plt
import numpy as np

data = parsed_data[-4:]
# Extract class names and values for r_acc and f_acc
classes = [item['name'] for item in data]
r_acc_values = [item['r_acc'] for item in data]
f_acc_values = [item['f_acc'] for item in data]

# Create bar chart
x = np.arange(len(classes))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.bar(x - width/2, r_acc_values, width, label='r_acc')
rects2 = ax.bar(x + width/2, f_acc_values, width, label='f_acc')

# Add some text for labels, title and axes ticks
ax.set_xlabel('Classes', fontsize=14)
ax.set_ylabel('Values', fontsize=14)
ax.set_title('(4)', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim(0, 1.2)  # Limit between 0 and 1.2 for better visualization
ax.tick_params(axis='both', which='major', labelsize=14)  # Set font size for both axes

# Show the plot
plt.show()















