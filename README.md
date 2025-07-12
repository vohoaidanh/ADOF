# ADOF: Image Synthesis Detection

<p align="center">
<img src="https://img.shields.io/aur/last-modified/google-chrome">
<img src="https://img.shields.io/badge/Author-Hoai--Danh.Vo-blue"> 
</p>

## Overview

ADOF is a deep learning framework for detecting AI-generated (synthetic) images. It supports training and evaluation on multiple datasets and provides tools for benchmarking and reproducibility.

---

## Demo

Try the online demo: [Demo](https://fcxyzypk72sf4hiappsy7se.streamlit.app/)

---

## Installation

> **Note:** If you are using Google Colab, all installation steps are already provided in the `train_notebook.ipynb` file included in this repository. You can simply open and run the notebook for a guided setup and usage experience.

1. **Clone the repository:**
   ```sh
   git clone https://github.com/vohoaidanh/ADOF.git
   cd ADOF
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   pip install tensorboardX
   ```

   *If you are using Google Colab or need extra dependencies:*
   ```sh
   pip install gdown regex imageio opencv-python scikit-learn scikit-image ftfy natsort blobfile
   ```


---

## Dataset Preparation

This project utilizes different datasets for training and evaluation. Please follow the instructions below to set up your data.

### Training Dataset

The primary training dataset is `ForenSynths`. The original download link for this dataset has expired. Therefore, a working Google Drive link for `ForenSynths_train` is provided directly within the `train_notebook.ipynb` file. Please refer to the notebook for instructions on downloading and preparing the training data.

### Evaluation Datasets

For evaluating the model, the following datasets are used. You can download and prepare them using the `download_dataset.sh` script.

**Automated Download:**
To automatically download and prepare all necessary **evaluation** datasets, run the `download_dataset.sh` script:

```sh
chmod +x download_dataset.sh
./download_dataset.sh
```

This script will download datasets like UniversalFakeDetect, GANGen-Detection, DiffusionForensics, AIGCDetect_testset, Diffusion1kStep, and CNN_synth_testset, and organize them into the `dataset` directory.

**Manual Download Links (for reference):**

| Dataset                | Paper/Source | Download Link |
|------------------------|--------------|---------------|
| ForenSynths (train/val/test) | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection) | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b) |
| GANGen-Detection       | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) | [Google Drive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing) |
| DiffusionForensics     | [DIRE ICCV2023](https://github.com/ZhendongWang6/DIRE) | [Google Drive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing) |
| UniversalFakeDetect    | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) | [Google Drive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing) |
| Diffusion1kStep        | -            | [Google Drive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing) |

### Directory Structure

Organize your datasets as shown below for the project to correctly locate them:

```
datasets/
└── ForenSynths_train/
    ├── train/
    │   ├── car/
    │   ├── cat/
    │   ├── chair/
    │   └── horse/
    ├── val/
    │   ├── car/
    │   ├── cat/
    │   ├── chair/
    │   └── horse/
    └── test/
        ├── biggan/
        ├── cyclegan/
        ├── deepfake/
        ├── gaugan/
        ├── progan/
        ├── stargan/
        ├── stylegan/
        └── stylegan2/
```

---

## Training

To train a model, run:

```sh
!find /datasets -type d -name "*ipynb*" -exec rm -r {} + #to remove folder .ipynb which make create dataset function crash
python train.py \
  --name adof-progan-4class \
  --dataroot ./datasets/ForenSynths_train \
  --classes car,cat,chair,horse \
  --batch_size 32 \
  --delr_freq 5 \
  --lr 0.0002 \
  --niter 30
```

**Other useful options:**
- `--num_thread 2` (number of data loading threads)

See `options/base_options.py` and `options/train_options.py` for all available arguments.

---

## Evaluation / Testing

To test a trained model:

```sh
python test.py \
  --model_path ./weights/ADOF_model_epoch_9.pth \
  --batch_size 32
```

- You may need to update the test set directory in `test.py` 

- For best results on AIGCDetectBenchmark, set `--no_resize` and `--no_crop` to `True`, and use `--batch_size 1`.

---

## Jupyter Notebook

You can also use `train_notebook.ipynb` for an interactive workflow, including installation, data download, training, and evaluation.

---

## Results

ADOF was extensively evaluated on public benchmarks and compared with state-of-the-art methods. The following tables summarize the results on the DiffusionForensics and Ojha datasets. ADOF achieves top or near-top performance in both accuracy and average precision (A.P.), demonstrating its effectiveness and generalizability.

### DiffusionForensics Results

| Method         | ADM Acc. | ADM A.P. | DDPM Acc. | DDPM A.P. | IDDPM Acc. | IDDPM A.P. | LDM Acc. | LDM A.P. | PNDM Acc. | PNDM A.P. | VQ-Diffusion Acc. | VQ-Diffusion A.P. | Stable Diff. v1 Acc. | Stable Diff. v1 A.P. | Stable Diff. v2 Acc. | Stable Diff. v2 A.P. | Mean Acc. | Mean A.P. |
|---------------|----------|----------|-----------|-----------|------------|------------|----------|----------|-----------|-----------|-------------------|-------------------|----------------------|----------------------|----------------------|----------------------|----------|----------|
| CNNDetection  | 53.9     | 71.8     | 62.7      | 76.6      | 50.2       | 82.7       | 50.4     | 78.7     | 50.8      | 90.3      | 50.0              | 71.0              | 38.0                 | 76.7                 | 52.0                 | 90.3                 | 51.0     | 79.8     |
| Frank         | 58.9     | 65.9     | 37.0      | 27.6      | 51.4       | 65.0       | 51.7     | 48.5     | 44.0      | 38.2      | 51.7              | 66.7              | 32.8                 | 52.3                 | 40.8                 | 37.5                 | 46.0     | 50.2     |
| Durall        | 39.8     | 42.1     | 52.9      | 49.8      | 55.3       | 56.7       | 43.1     | 39.9     | 44.5      | 47.3      | 38.6              | 38.3              | 39.5                 | 56.3                 | 62.1                 | 55.8                 | 47.0     | 48.3     |
| Patchfor      | 77.5     | 93.9     | 62.3      | 97.1      | 50.0       | 91.6       | 99.5     | 100.0    | 50.2      | 99.9      | 100.0             | 100.0             | 90.7                 | 99.8                 | 94.8                 | 100.0                | 78.1     | 97.8     |
| F3Net         | 80.9     | 96.9     | 84.7      | 99.4      | 74.7       | 98.9       | 100.0    | 100.0    | 72.8      | 99.5      | 100.0             | 100.0             | 73.4                 | 97.2                 | 99.8                 | 100.0                | 85.8     | 99.0     |
| SelfBland     | 57.0     | 59.0     | 61.9      | 49.6      | 63.2       | 66.9       | 83.3     | 92.2     | 48.2      | 48.2      | 77.2              | 82.7              | 46.2                 | 68.0                 | 71.2                 | 73.9                 | 63.5     | 67.6     |
| GANDetection  | 51.1     | 53.1     | 62.3      | 46.4      | 50.2       | 63.0       | 51.6     | 48.1     | 50.6      | 79.0      | 51.1              | 51.2              | 39.8                 | 65.6                 | 50.1                 | 36.9                 | 50.8     | 55.4     |
| LGrad         | 86.4     | 97.5     | 99.9      | 100.0     | 66.1       | 92.8       | 99.7     | 100.0    | 69.5      | 98.5      | 96.2              | 100.0             | 90.4                 | 99.4                 | 97.1                 | 100.0                | 88.2     | 98.5     |
| Ojha          | 78.4     | 92.1     | 72.9      | 78.8      | 75.0       | 92.8       | 82.2     | 97.1     | 75.3      | 92.5      | 83.5              | 97.7              | 56.4                 | 90.4                 | 71.5                 | 92.4                 | 74.4     | 91.7     |
| NPR           | 88.6     | 98.9     | 99.8      | 100.0     | 91.8       | 99.8       | 100.0    | 100.0    | 91.2      | 100.0     | 100.0             | 100.0             | 97.4                 | 99.8                 | 93.8                 | 100.0                | 95.3     | 99.8     |
| **ADOF(ours)**| **93.5** | **99.0** | 99.6      | **100.0** | **99.2**   | **100.0**  | 99.9     | **100.0**| **97.4**  | 99.9      | 97.1              | 99.8              | **99.8**             | **100.0**            | **99.9**             | **100.0**            | **98.3** | **99.8** |
 

### Ojha Dataset Results

| Method         | DALLE Acc. | DALLE A.P. | Glide(100_10) Acc. | Glide(100_10) A.P. | Glide(100_27) Acc. | Glide(100_27) A.P. | Glide(50_27) Acc. | Glide(50_27) A.P. | ADM Acc. | ADM A.P. | LDM(100) Acc. | LDM(100) A.P. | LDM(200) Acc. | LDM(200) A.P. | LDM(200_cfg) Acc. | LDM(200_cfg) A.P. | Mean Acc. | Mean A.P. |
|---------------|-----------|------------|--------------------|--------------------|--------------------|--------------------|-------------------|-------------------|----------|----------|--------------|--------------|--------------|--------------|-------------------|-------------------|----------|----------|
| CNNDetection  | 51.8      | 61.3       | 53.3               | 72.9               | 53.0               | 71.3               | 54.2              | 76.0              | 54.9     | 66.6     | 51.9         | 63.7         | 52.0         | 64.5         | 51.6              | 63.1              | 52.8     | 67.4     |
| Frank         | 57.0      | 62.5       | 53.6               | 44.3               | 50.4               | 40.8               | 52.0              | 42.3              | 53.4     | 52.5     | 56.6         | 51.3         | 56.4         | 50.9         | 56.5              | 52.1              | 54.5     | 49.6     |
| Durall        | 55.9      | 58.0       | 54.9               | 52.3               | 48.9               | 46.9               | 51.7              | 49.9              | 40.6     | 42.3     | 62.0         | 62.6         | 61.7         | 61.7         | 58.4              | 58.5              | 54.3     | 54.0     |
| Patchfor      | 79.8      | 99.1       | 87.3               | 99.7               | 82.8               | 99.1               | 84.9              | 98.8              | 74.2     | 81.4     | 95.8         | 99.8         | 95.6         | 99.9         | 94.0              | 99.8              | 86.8     | 97.2     |
| F3Net         | 71.6      | 79.9       | 88.3               | 95.4               | 87.0               | 94.5               | 88.5              | 95.4              | 69.2     | 70.8     | 74.1         | 84.0         | 73.4         | 83.3         | 80.7              | 89.1              | 79.1     | 86.5     |
| SelfBland     | 52.4      | 51.6       | 58.8               | 63.2               | 59.4               | 64.1               | 64.2              | 68.3              | 58.3     | 63.4     | 53.0         | 54.0         | 52.6         | 51.9         | 51.9              | 52.6              | 56.3     | 58.7     |
| GANDetection  | 67.2      | 83.0       | 51.2               | 52.6               | 51.1               | 51.9               | 51.7              | 53.5              | 49.6     | 49.0     | 54.7         | 65.8         | 54.9         | 65.9         | 53.8              | 58.9              | 54.3     | 60.1     |
| LGrad         | 88.5      | 97.3       | 89.4               | 94.9               | 87.4               | 93.2               | 90.7              | 95.1              | **86.6** | **100.0**| 94.8         | 99.2         | 94.2         | 99.1         | 95.9              | 99.2              | 90.9     | 97.2     |
| Ojha          | 89.5      | 96.8       | 90.1               | 97.0               | 90.7               | 97.2               | 91.1              | 97.4              | 75.7     | 85.1     | 90.5         | 97.0         | 90.2         | 97.1         | 77.3              | 88.6              | 86.9     | 94.5     |
| NPR           | 94.5      | **99.5**   | 98.2               | 99.8               | 97.8               | 99.7               | 98.2              | 99.8              | 75.8     | 81.0     | **99.3**     | 99.9         | **99.1**     | 99.9         | **99.0**          | 99.8              | **95.2** | 97.4     |
| RINE          | **95.0**  | 99.5       | 90.7               | 99.2               | 88.9               | 99.1               | 92.6              | 99.5              | 76.1     | 96.6     | 98.7         | 99.9         | 98.3         | 99.9         | 88.2              | 98.7              | 91.1     | 99.0     |
| **ADOF(ours)**| 92.1      | 98.3       | **98.6**           | **100.0**          | **98.7**           | **100.0**          | **98.4**          | **99.9**          | 75.9     | 87.6     | 98.8         | **100.0**    | 98.6         | **99.9**     | 98.5              | **99.9**          | 94.9     | **98.2** |



### Model Size, Speed, and Efficiency Comparison (DiffusionForensics)

The table below compares the number of parameters, processing time, inference time, FLOPs, and mean accuracy/average precision (acc/ap) of ADOF and other methods on the DiffusionForensics dataset. ADOF achieves state-of-the-art performance with a much smaller and faster model.

| Method                | Parameters         | Processing (ms) | Inference Time (ms) | FLOPs              | Means (acc/ap) |
|-----------------------|-------------------|-----------------|---------------------|--------------------|----------------|
| LGrad                 | 25.56 × 10⁶       | 11.6            | 4.81                | 4.12 × 10⁹         | 88.2 / 98.5    |
| DIRE†                 | 25.56 × 10⁶       | 4,502.7         | 4.81                | 4.12 × 10⁹         | 97.9 / **100** |
| Ojha                  | 427.62 × 10⁶      | None            | 29.19               | 77.83 × 10⁹        | 74.4 / 91.7    |
| **ADOF (ours)**       | **1.44 × 10⁶**    | **0.40**        | **2.43**            | **1.74 × 10⁹**     | **98.3 / 99.8**|

## Acknowledgments

This repository borrows partially from [NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).

---

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  
You may use, share, and adapt this code **for non-commercial research and academic purposes only**. Commercial use is strictly prohibited.

🔗 [CC BY-NC 4.0 Legal Code](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Citation
If you use this code or its concepts in your research, please cite the following publication:

**Minimalist Preprocessing Approach for Image Synthesis Detection**  
Hoai-Danh Vo, Trung-Nghia Le  
*Information and Communication Technology (SOICT 2024), Communications in Computer and Information Science, vol 2350, Springer, Singapore, 2025.*  
[https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8](https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8)

```bibtex
@inproceedings{Vo2025MinimalistPA,
  author={Vo, Hoai-Danh and Le, Trung-Nghia},
  booktitle={Information and Communication Technology},
  doi={10.1007/978-981-96-4282-3_8},
  editor={Buntine, Wray and Fjeld, Morten and Tran, Truyen and Tran, Minh-Triet and Huynh Thi Thanh, Binh and Miyoshi, Takumi},
  isbn={978-981-96-4281-6},
  pages={88-99},
  publisher={Springer Nature Singapore},
  title={Minimalist Preprocessing Approach for Image Synthesis Detection},
  year={2025}
}
```

---

## Related Publication

This section provides additional details about the academic paper that describes the core methodology of this project.

**Minimalist Preprocessing Approach for Image Synthesis Detection**  
Hoai-Danh Vo, Trung-Nghia Le  
*Information and Communication Technology (SOICT 2024), Communications in Computer and Information Science, vol 2350, Springer, Singapore, 2025.*  
[https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8](https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8)

**Abstract:**
> Generative models have significantly advanced image generation, resulting in synthesized images that are increasingly indistinguishable from authentic ones. However, the creation of fake images with malicious intent is a growing concern. In this paper, we introduce a simple yet efficient method that captures pixel fluctuations between neighboring pixels by calculating the gradient, which highlights variations in grayscale intensity. This approach functions as a high-pass filter, emphasizing key features for accurate image distinction while minimizing color influence. Our experiments on multiple datasets demonstrate that our method achieves accuracy levels comparable to state-of-the-art techniques while requiring minimal computational resources. Therefore, it is suitable for deployment on low-end devices such as smartphones.

**Experimental Results:**
> The proposed method was evaluated on several public datasets for image synthesis detection. It achieved accuracy comparable to state-of-the-art deep learning approaches, while requiring significantly less computational resources. This makes the method especially suitable for deployment on low-end devices such as smartphones and embedded systems. The results confirm the effectiveness and efficiency of the minimalist preprocessing approach for real-world applications.

