import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    data_loader = create_dataloader(opt)
    if opt.mode == 'custom':
        with torch.no_grad():
            y_true, y_pred = [], []
            for img, label in data_loader:
                in_tens = img.cuda()
                output = model(in_tens)  # Logits của mô hình, có thể có shape [batch_size, num_classes]
                
                # Áp dụng softmax để tính xác suất cho mỗi lớp
                probs = torch.softmax(output, dim=1)
                
                # Dự đoán lớp với xác suất cao nhất
                preds = probs.argmax(dim=1)
                
                y_pred.extend(preds.tolist())  # Dự đoán cuối cùng
                y_true.extend(label.flatten().tolist())  # Nhãn thực tế
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Tính accuracy cho từng lớp
        acc = accuracy_score(y_true, y_pred)
        
        # Tính accuracy cho từng lớp riêng biệt (r_acc cho lớp 0, f_acc cho lớp 1, và acc cho lớp 2)
        class_accuracies = {}
        for i in range(3):
            class_mask = (y_true == i)
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            class_accuracies[f'class_{i}_acc'] = class_acc
        
        # Tính average precision score cho từng lớp
        #ap = []
        #for i in range(3):
        #    ap_class = average_precision_score((y_true == i).astype(int), probs[:, i].cpu())
        #    ap.append(ap_class)
        
        return acc, 0, class_accuracies, 0, y_true, y_pred
    
    else:  
        with torch.no_grad():
            y_true, y_pred = [], []
            for img, label in data_loader:
                in_tens = img.cuda()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())
    
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
