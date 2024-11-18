import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
FONT_SIZE = 40

# Dữ liệu giả lập
methods = ['Ours', 'Ojha', 'LGrad', 'RINE']
acc = [94.9, 86.9, 90.9, 91.1]  # accuracy in percentage
flops = [1.74, 77.83, 4.12, 77.8]  # FLOPs
model_size = [1.4, 472.6, 25.6, 433.9]  # Model size (độ lớn mô hình, giả sử là MB)
colors = ['#FF6347', '#FFA500', '#00FFFF', '#90EE90']  # Màu sắc tùy chỉnh cho mỗi phương pháp


# Chuyển FLOPs sang log scale
log_flops = flops#np.log10(flops)

# Tạo figure và axis với kích thước lớn
fig, ax = plt.subplots(figsize=(28, 20))  # Tăng kích thước cho poster

# Vẽ scatter plot
scatter = ax.scatter(log_flops, acc, s=np.array(model_size)*1000, c=colors, alpha=1, zorder=5)

# Thêm chú thích cho từng điểm (nếu muốn, có thể sử dụng model_size hoặc các thông tin khác)
for i, size in enumerate(model_size):
    ax.annotate(
        f"{size}M", 
        (log_flops[i], acc[i]),   # Vị trí gốc của chú thích
        fontsize=FONT_SIZE, 
        color='black',
        xytext=(0, -35),  # Điều chỉnh vị trí văn bản, có thể là tuple (x_offset, y_offset)
        textcoords='offset points',  # Hệ tọa độ tính toán từ điểm gốc (log_flops[i], acc[i])
        ha='center',   # Căn chỉnh văn bản theo chiều ngang (center, left, right)
        va='center',   # Căn chỉnh văn bản theo chiều dọc (center, top, bottom)
        zorder=6 
    )

# Thêm tên phương pháp vào các điểm dữ liệu
#for i, method in enumerate(methods):
#    ax.text(log_flops[i], acc[i], method, fontsize=14, ha='right', va='bottom')

# Thiết lập trục tung và trục hoành
ax.set_xlabel('FLOPs (log scale)', fontsize=FONT_SIZE*1.25, fontweight='bold')  # Tăng kích thước font chữ
ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE*1.25, fontweight='bold')
ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=FONT_SIZE, pad=22)  # Thiết lập kích thước font chữ cho xtick
ax.tick_params(axis='y', labelsize=FONT_SIZE, pad=22)  # Thiết lập kích thước font chữ cho xtick

# Thiết lập giới hạn trục Y
ax.set_ylim(82, 96)  # Giới hạn trục Y từ 70% đến 100%

# Thiết lập giới hạn trục X tự động
min_flops = min(log_flops)
max_flops = max(log_flops)
ax.set_xlim(0 , 800)  # Tự động co lại trục x chỉ với các giá trị có data

# Thiết lập lưới với nét đứt
#ax.set_axisbelow(True)  # Đặt lưới dưới các điểm dữ liệu
#ax.xaxis.set_tick_params(labelleft=False)  # Tắt nhãn tick trên trục Y

custom_xticks = [2, 78, 4]  # Ví dụ các giá trị FLOPs tùy chọn
custom_labels = ["2x$10^9$", "78x$10^9$", "4x$10^9$"]
  # Các nhãn tùy chỉnh

# Thiết lập các xtick và nhãn tương ứng
ax.set_xticks(custom_xticks)  # Đặt vị trí xtick
ax.set_xticklabels(custom_labels, fontsize=FONT_SIZE)  # Đặt nhãn tùy chỉnh với kích thước font chữ

# Tạo custom legend với các chấm tròn
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=40, label=method)
                   for method, color in zip(methods, colors)]

# Thêm legend với các chấm tròn màu sắc
ax.legend(handles=legend_elements, fontsize=FONT_SIZE*1.1, labelspacing=1.1)
# Tắt các tick label tự động
ax.grid(True)  # Bật lưới trên biểu đồ
#ax.set_xticklabels([])

# Vẽ lưới chỉ tại các điểm dữ liệu
#for i in range(len(log_flops)):
#    ax.vlines(log_flops[i], 0, 100, color='gray', linestyles='-', alpha=0.2)  # Đường dọc tại từng điểm
    #ax.hlines(acc[i], min_flops - 0.1, log_flops[i], color='gray', linestyles='dotted', alpha=0.5)  # Đường ngang tại từng điểm

# Hiển thị biểu đồ
plt.tight_layout()  # Đảm bảo tất cả các phần tử của biểu đồ không bị cắt khi in
plt.show()









