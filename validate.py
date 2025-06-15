import torch
import numpy as np
from networks.resnet import resnet50
from options.test_options import TestOptions
from data import create_dataloader

from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torchvision.transforms.functional as TF
from PIL import Image

from util import delete_dot_folders

# def validate(model, opt):
#     data_loader = create_dataloader(opt)

#     with torch.no_grad():
#         y_true, y_pred = [], []
#         for img, label in data_loader:
#             in_tens = img.cuda()
#             y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
#             y_true.extend(label.flatten().tolist())

#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
#     f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
#     acc = accuracy_score(y_true, y_pred > 0.5)
#     ap = average_precision_score(y_true, y_pred)
#     return acc, ap, r_acc, f_acc, y_true, y_pred

def validate(model, opt, return_images=False):
    data_loader = create_dataloader(opt)
    device = next(model.parameters()).device
    y_true, y_pred, y_score = [], [], []
    misclassified_images = []
    
    count = 0;
    with torch.no_grad():
        for img, label in data_loader:
            count+=1
            in_tens = img.to(device)
            output = model(in_tens).sigmoid().flatten()
            pred = (output > 0.5).int().cpu().tolist()
            label_flat = label.flatten().tolist()

            y_pred.extend(pred)
            y_score.extend(output.cpu().tolist())
            y_true.extend(label_flat)

            if return_images:  # thu thập hình ảnh bị phân loại sai
                for i in range(len(pred)):
                    if pred[i] != label_flat[i]:
                        # Chuyển từ Tensor sang PIL Image (giả sử ảnh có shape [C, H, W])
                        img_pil = TF.to_pil_image(img[i].cpu())
                        misclassified_images.append((img_pil, pred[i], label_flat[i]))
            

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = accuracy_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_score)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    conf_matrix = confusion_matrix(y_true, y_pred)

    results = {
        "acc": acc,
        "ap": ap,
        "r_acc": r_acc,
        "f_acc": f_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": conf_matrix,
    }

    if return_images:
        return results, misclassified_images
    else:
        return results

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    opt.dataroot = r"C:\Users\danhv\Downloads\ForenSynths_test_200_1"
    opt.num_threads = 0
    opt.classes = []
    model = resnet50(num_classes=1)
    state_dict = torch.load("weights/ADOF_model_epoch_9.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    # model.cuda()
    model.eval()

    delete_dot_folders(opt.dataroot)
    results = validate(model, opt)
    print(results)
    
    
    # print("accuracy:", acc)
    # print("average precision:", avg_precision)

    # print("accuracy of real images:", r_acc)
    # print("accuracy of fake images:", f_acc)
