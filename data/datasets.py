import os
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice, randint
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

# =============================================================================
#     dset = datasets.ImageFolder(
#             root,
#             transforms.Compose([
#                 rz_func,
#                 transforms.Lambda(lambda img: data_augment(img, opt)),
#                 crop_func,
#                 flip_func,
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]))
#     return dset
# =============================================================================
    dset = RealImageDataset(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path

class RealImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, data_augment=None, opt=None):
        """
        :param root_dir: Thư mục chứa dữ liệu
        :param transform: Các phép biến đổi như resize, normalization, v.v.
        :param data_augment: Hàm tùy chỉnh cho data augmentation (nếu có)
        :param opt: Các tham số khác có thể được truyền vào hàm data_augment
        Label == 0 nghĩa là 2 hình ảnh giống nhau, Label == 1 Nghĩa là 2 hình ảnh khác nhau
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_augment = data_augment
        self.opt = opt

        # Đọc các ảnh từ các lớp 0_real và 1_fake
        self.real_images = []
        self.fake_images = []

        # Đọc thư mục chứa ảnh từ class 0_real
        real_dir = os.path.join(root_dir, "0_real")
        if os.path.isdir(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.real_images.append(os.path.join(real_dir, img_name))
        
        # Đọc thư mục chứa ảnh từ class 1_fake
#        fake_dir = os.path.join(root_dir, "1_fake")
#        if os.path.isdir(fake_dir):
#            for img_name in os.listdir(fake_dir):
#                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                    self.fake_images.append(os.path.join(fake_dir, img_name))
        
        # Kiểm tra nếu có đủ ảnh trong cả 2 lớp
#        if len(self.real_images) == 0 or len(self.fake_images) == 0:
#            raise ValueError("Không tìm thấy ảnh trong một hoặc cả hai lớp (0_real, 1_fake).")

    def __len__(self):
        """Trả về số lượng mẫu trong dataset."""
        return len(self.real_images)

    def __getitem__(self, idx):
        """Trả về 2 ảnh, một từ lớp 0_real và một từ lớp 1_fake."""
        # Chọn ngẫu nhiên ảnh từ lớp 0_real
        label = choice([0, 1])
        real_img_path = self.real_images[idx]
        real_img = Image.open(real_img_path).convert('RGB')
        
        if label == 0:
            # Label = 0 nghĩa là 2 hình ảnh giống nhau, Label = 1 Nghĩa là 2 hình ảnh khác nhau
            # Áp dụng data augmentation nếu có
            if self.data_augment:
                real_img = self.data_augment(real_img, self.opt)

            # Áp dụng các phép biến đổi (resize, crop, flip, normalize)
            if self.transform:
                real_img = self.transform(real_img)
            
            real_img2 = real_img
            
        else:
            idx2 = random_exclude([idx], low=0, high=len(self.real_images))
            img_path = self.real_images[idx2]
            real_img2 = Image.open(img_path).convert('RGB')
            
            # Áp dụng data augmentation nếu có
            if self.data_augment:
                real_img = self.data_augment(real_img, self.opt)
                real_img2 = self.data_augment(real_img2, self.opt)

            # Áp dụng các phép biến đổi (resize, crop, flip, normalize)
            if self.transform:
                real_img = self.transform(real_img)
                real_img2 = self.transform(real_img2)

        
        # Trả về cả hai ảnh và nhãn tương ứng
        return (real_img, real_img2) , label

def random_exclude(exclude, low, high):
    while True:
        value = randint(low, high - 1)
        if value not in exclude:
            return value

def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])

if __name__ == '__main__':
    from options.train_options import TrainOptions
    from data import create_dataloader
    import torch
    def get_dataset(opt):
        classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
        print(classes)
        if '0_real' not in classes or '1_fake' not in classes:
            dset_lst = []
            for cls in classes:
                root = opt.dataroot + '/' + cls
                dset = dataset_folder(opt, root)
                dset_lst.append(dset)
            return torch.utils.data.ConcatDataset(dset_lst)
        return dataset_folder(opt, opt.dataroot)

    opt = TrainOptions().parse()
    opt.dataroot = r"D:/Downloads/dataset/progan_val_4_class/train"
    opt.classes = ['car', 'cat', 'chair', 'horse']
    opt.num_threads = 0
    dataset = get_dataset(opt)
    data_loader = create_dataloader(opt)
    len(dataset)
    dataiter = iter(dataset)
    sample = next(dataiter)
    print(sample[1])
    
    for i in data_loader:
        print(i[1].shape)
        
    

    