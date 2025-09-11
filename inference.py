import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from networks.resnet import resnet50
import csv
import argparse
import random, numpy as np

# --------------------------
# Fix random seed
# --------------------------
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

# --------------------------
# Dataset để load ảnh từ folder
# --------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        for r, _, files in os.walk(root):
            for f in files:
                if any(f.lower().endswith(ext) for ext in exts):
                    self.samples.append(os.path.join(r, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

# --------------------------
# Main inference
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference on image folder")
    parser.add_argument("--test_root", type=str, required=True, help="Folder chứa ảnh test")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint .pth đã train")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="File CSV output")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng phân loại (default=0.5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size khi inference")
    args = parser.parse_args()

    seed_torch(100)

    # --- Load model ---
    model = resnet50(num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location="cuda"))
    model.cuda()
    model.eval()

    # --- Transform ảnh ---
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # --- Dataset + DataLoader ---
    dataset = ImageFolderDataset(args.test_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Inference ---
    results = []
    with torch.no_grad():
        for imgs, paths in dataloader:
            imgs = imgs.cuda()
            probs = torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy()
            for path, prob in zip(paths, probs):
                label = "fake" if prob > args.threshold else "real"
                results.append([path, float(prob), label])
                print(f"{path} -> {label} (prob={prob:.4f})")

    # --- Xuất CSV ---
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Probability", "Prediction"])
        writer.writerows(results)

    print(f"Inference hoàn tất, lưu kết quả tại {args.output_csv}")
