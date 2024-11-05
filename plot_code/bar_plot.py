
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import json
# Define regex pattern to extract all the data columns
pattern = r"\((\d+)\s+(\w+)\s+\)\s+acc:\s([\d.]+);\s+ap:\s([\d.]+);\s+r_acc:\s([\d.]+);\s+f_acc:\s([\d.]+)"

file_path = r"D:\K32\do_an_tot_nghiep\THESIS\Material\origin-resnet18-carReal-horseFake.txt" # Thay thế bằng đường dẫn tới file của bạn
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
#print(json_data)
IDS = 3
MAX = parsed_data[4:8]
F1 = 2*MAX[0]['f_acc']*MAX[0]['r_acc']/(MAX[0]['f_acc']+MAX[0]['r_acc'])
F3 = 2*MAX[IDS]['f_acc']*MAX[IDS]['r_acc']/(MAX[IDS]['f_acc']+MAX[IDS]['r_acc'])
MEAN = 0.5*(F1+F3)
ID = 0
for i in range(0, len(parsed_data), 4):
    if i==0:
        continue
    
    Sample = parsed_data[i:i+4]
    
    F1 = 2*Sample[0]['f_acc']*Sample[0]['r_acc']/(Sample[0]['f_acc']+Sample[0]['r_acc'])
    F3 = 2*Sample[IDS]['f_acc']*Sample[IDS]['r_acc']/(Sample[IDS]['f_acc']+Sample[IDS]['r_acc'])
    m = 0.5*(F1+F3)
    if m > MEAN:
        MAX = Sample


import matplotlib.pyplot as plt
import numpy as np


#data = []
#data.insert(0,MAX[0])
#data.append(MAX[0]) #parsed_data[ID:ID+4]

print(data)
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
ax.set_title('(1)', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim(0, 1.2)  # Limit between 0 and 1.2 for better visualization
ax.tick_params(axis='both', which='major', labelsize=14)  # Set font size for both axes
data1 = []
# Show the plot
plt.show()
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

"""
  name    acc       ap  r_acc  f_acc
0  car  51.20  80.8000   1.00  0.000
1  car  52.00  83.6848   1.00  0.040
2  car  66.00  95.2356   1.00  0.320
3  car  69.00  96.1275   1.00  0.380
4  car  75.25  97.9016   1.00  0.505
5  car  78.50  95.8671   0.98  0.590
"""



# Convert data to DataFrame for plotting
df = pd.DataFrame(data1)


df['f1_score'] = [0.0, 0.0769, 0.4848, 0.5507, 0.6711, 0.7577]

plt.figure(figsize=(14, 8))
percent_labels = ["0%", "50%", "65%", "80%", "95%", "98%"]

plt.plot(percent_labels, df['f1_score'], color='b', marker='o', markersize=12, label='Car', linewidth=2.5)

# Adjusting text size
#plt.title('F1 score (%) for Car', fontsize=16)
plt.xlabel('Low-Frequency Cutoff (%)', fontsize=18)
plt.ylabel('F1 score (%)', fontsize=18)
plt.ylim(0, 1)  # Đặt giới hạn cho trục y từ 0 đến 100
plt.grid(True, linestyle='--')
plt.tick_params(axis='both', labelsize=16) 
plt.legend(fontsize=18)  # Kích thước font cho chú thích

# Saving to JSON file
file_path = 'car_data.json'
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)




