# ADOF

<p align="center">
	<br>
	---
</p>

<img src="./NPR.png" width="100%" alt="overall pipeline">

#Reference github repository for the paper [---](url).
```
@misc{...
}
```
## Environment setup
**Classification environment:** 
We recommend installing the required packages by running the command:
```sh
pip install -r requirements.txt
```
In order to ensure the reproducibility of the results, we provide the following suggestions：
- Docker image: nvcr.io/nvidia/tensorflow:21.02-tf1-py3
- Conda environment: [./pytorch18/bin/python](https://drive.google.com/file/d/16MK7KnPebBZx5yeN6jqJ49k7VWbEYQPr/view) 
- Random seed during testing period: [Random seed](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/b4e1bfa59ec58542ab5b1e78a3b75b54df67f3b8/test.py#L14)

## Getting the data
<!-- 
Download dataset from [CNNDetection CVPR2020 (Table1 results)](https://github.com/peterwang512/CNNDetection), [GANGen-Detection (Table2 results)](https://github.com/chuangchuangtan/GANGen-Detection) ([googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)), [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) ([googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=drive_link)), [DIRE 2023ICCV](https://github.com/ZhendongWang6/DIRE) ([googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)), Diffusion1kStep [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing).
-->
|                        | paper  | Url  |
|:----------------------:|:-----:|:-----:|
| Train set              | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Val   set              | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Table1 Test            | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)                   | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                 | 
| Table2 Test            | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)        | [googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)   | 
| Table3 Test            | [DIRE ICCV2023](https://github.com/ZhendongWang6/DIRE)                                  | [googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)   | 
| Table4 Test            | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect)        | [googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing)| 
| Table5 Test            | Diffusion1kStep                                                                         | [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing)   | 

```
pip install gdown==4.7.1

chmod 777 ./download_dataset.sh

./download_dataset.sh
```
## Directory structure
<details>
<summary> Click to expand the folder tree structure. </summary>

```
datasets
|-- ForenSynths_train_val
|   |-- train
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   `-- horse
|   `-- val
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   `-- horse
|   |-- test
|       |-- biggan
|       |-- cyclegan
|       |-- deepfake
|       |-- gaugan
|       |-- progan
|       |-- stargan
|       |-- stylegan
|       `-- stylegan2
`-- Generalization_Test
    |-- ForenSynths_test       # Table1
    |   |-- biggan
    |   |-- cyclegan
    |   |-- deepfake
    |   |-- gaugan
    |   |-- progan
    |   |-- stargan
    |   |-- stylegan
    |   `-- stylegan2
    |-- GANGen-Detection     # Table2
    |   |-- AttGAN
    |   |-- BEGAN
    |   |-- CramerGAN
    |   |-- InfoMaxGAN
    |   |-- MMDGAN
    |   |-- RelGAN
    |   |-- S3GAN
    |   |-- SNGAN
    |   `-- STGAN
    |-- DiffusionForensics  # Table3
    |   |-- adm
    |   |-- ddpm
    |   |-- iddpm
    |   |-- ldm
    |   |-- pndm
    |   |-- sdv1_new
    |   |-- sdv2
    |   `-- vqdiffusion
    `-- UniversalFakeDetect # Table4
    |   |-- dalle
    |   |-- glide_100_10
    |   |-- glide_100_27
    |   |-- glide_50_27
    |   |-- guided          # Also known as ADM.
    |   |-- ldm_100
    |   |-- ldm_200
    |   `-- ldm_200_cfg
    |-- Diffusion1kStep     # Table5
        |-- DALLE
        |-- ddpm
        |-- guided-diffusion    # Also known as ADM.
        |-- improved-diffusion  # Also known as IDDPM.
        `-- midjourney


```
</details>

## Training the model 
```sh
CUDA_VISIBLE_DEVICES=0 ./pytorch18/bin/python train.py --name 4class-resnet-car-cat-chair-horse --dataroot ./datasets/ForenSynths_train_val --classes car,cat,chair,horse --batch_size 32 --delr_freq 5 --lr 0.0002 --niter 50
```

## Testing the detector
Modify the dataroot in test.py.
```sh
CUDA_VISIBLE_DEVICES=0 ./pytorch18/bin/python test.py --model_path ./weights/ADOF_model_epoch_9.pth  --batch_size {BS}
```

## Detection Results

### [AIGCDetectBenchmark](https://drive.google.com/drive/folders/1p4ewuAo7d5LbNJ4cKyh10Xl9Fg2yoFOw) using [ProGAN-4class checkpoint](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/main/model_epoch_last_3090.pth)

When testing on AIGCDetectBenchmark, set no_resize and no_crop to True, and set batch_size to 1.
To deal with images of odd sizes, add the following code in [network/resnet.py](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/e2dbbe673c69c0c7237726e809a725a0308ec43d/networks/resnet.py#L163).
```
n,c,w,h = x.shape
if w%2 == 1 : x = x[:,:,:-1,:]
if h%2 == 1 : x = x[:,:,:,:-1]
```

| Generator   |  CNNSpot | FreDect |   Fusing  | GramNet |   LNP   |  LGrad  |  DIRE-G | DIRE-D |  UnivFD |  RPTCon | NPR  |
|  :---------:| :-----:  |:-------:| :--------:|:-------:|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:----:|
| ProGAN      |  100.00  |  99.36  |   100.00  |  99.99  |  99.67  |  99.83  |  95.19  |  52.75 |  99.81  |  100.00 | 99.9 |
| StyleGan    |  90.17   |  78.02  |   85.20   |  87.05  |  91.75  |  91.08  |  83.03  |  51.31 |  84.93  |  92.77  | 96.1 |
| BigGAN      |  71.17   |  81.97  |   77.40   |  67.33  |  77.75  |  85.62  |  70.12  |  49.70 |  95.08  |  95.80  | 87.3 |
| CycleGAN    |  87.62   |  78.77  |   87.00   |  86.07  |  84.10  |  86.94  |  74.19  |  49.58 |  98.33  |  70.17  | 90.3 |
| StarGAN     |  94.60   |  94.62  |   97.00   |  95.05  |  99.92  |  99.27  |  95.47  |  46.72 |  95.75  |  99.97  | 99.6 |
| GauGAN      |  81.42   |  80.57  |   77.00   |  69.35  |  75.39  |  78.46  |  67.79  |  51.23 |  99.47  |  71.58  | 85.4 |
| Stylegan2   |  86.91   |  66.19  |   83.30   |  87.28  |  94.64  |  85.32  |  75.31  |  51.72 |  74.96  |  89.55  | 98.1 |
| WFIR        |  91.65   |  50.75  |   66.80   |  86.80  |  70.85  |  55.70  |  58.05  |  53.30 |  86.90  |  85.80  | 60.7 |
| ADM         |  60.39   |  63.42  |   49.00   |  58.61  |  84.73  |  67.15  |  75.78  |  98.25 |  66.87  |  82.17  | 84.9 |
| Glide       |  58.07   |  54.13  |   57.20   |  54.50  |  80.52  |  66.11  |  71.75  |  92.42 |  62.46  |  83.79  | 96.7 |
| Midjourney  |  51.39   |  45.87  |   52.20   |  50.02  |  65.55  |  65.35  |  58.01  |  89.45 |  56.13  |  90.12  | 92.6 |
| SDv1.4      |  50.57   |  38.79  |   51.00   |  51.70  |  85.55  |  63.02  |  49.74  |  91.24 |  63.66  |  95.38  | 97.4 |
| SDv1.5      |  50.53   |  39.21  |   51.40   |  52.16  |  85.67  |  63.67  |  49.83  |  91.63 |  63.49  |  95.30  | 97.5 |
| VQDM        |  56.46   |  77.80  |   55.10   |  52.86  |  74.46  |  72.99  |  53.68  |  91.90 |  85.31  |  88.91  | 90.1 |
| Wukong      |  51.03   |  40.30  |   51.70   |  50.76  |  82.06  |  59.55  |  54.46  |  90.90 |  70.93  |  91.07  | 91.7 |
| DALLE2      |  50.45   |  34.70  |   52.80   |  49.25  |  88.75  |  65.45  |  66.48  |  92.45 |  50.75  |  96.60  | 99.6 |
| Average     |  70.78   |  64.03  |   68.38   |  68.67  |  83.84  |  75.34  |  68.68  |  71.53 |  78.43  |  89.31  | **91.7** |


## Acknowledgments

This repository borrows partially from the [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).
