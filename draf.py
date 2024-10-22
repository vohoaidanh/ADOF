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
plt.plot(x_vals, y_vals, label='Đa thức nội suy .')
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



labels = torch.tensor([1, 0, 0, 1,1,1,0,1,0,0,1])


labels1 = (labels == 1).nonzero(as_tuple=True)[0]


















