import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
from networks.model import build_model
import json

import numpy as np
import random
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
                'ForenSynths': { 'dataroot'   : '/workspace/datasets/ForenSynths/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

           'GANGen-Detection': { 'dataroot'   : '/workspace/dataset/GANGen-Detection/',
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },

         'DiffusionForensics': { 'dataroot'   : '/workspace/dataset/DiffusionForensics/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

        'UniversalFakeDetect': { 'dataroot'   : '/workspace/dataset/UniversalFakeDetect/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

                 }


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
#model = resnet50(num_classes=1)
model = build_model(backbone=opt.backbone, num_features=opt.num_features, pretrained=False, num_classes=1, freeze_exclude=None)

if torch.cuda.is_available() and len(opt.gpu_ids) > 1:
    model = torch.nn.DataParallel(model)  # Sử dụng DataParallel
# =============================================================================
# def remove_module_prefix(state_dict):
#     #Remove unwanted name in state_dict encoutered when training model with multi GPU
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         # Xóa tiền tố 'module.' nếu có
#         new_key = k.replace('module.', '')  
#         new_state_dict[new_key] = v
#     return new_state_dict
# =============================================================================

state_dict = torch.load(opt.model_path, map_location='cpu')
if 'model' in state_dict.keys():
    state_dict = state_dict['model']
    
model.load_state_dict(state_dict, strict=True)
model.cuda()
model.eval()
log_data = {}  # Tạo dictionary để lưu log

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = [];aps = []
    # Tạo một list để lưu log cho từng mẫu
    test_results = []

    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
        accs.append(acc);aps.append(ap)

        # Lưu kết quả vào dictionary
        result = {
            'id': v_id,
            'val': val,
            'acc': acc,
            'ap': ap,
            'r_acc': r_acc,
            'f_acc': f_acc
        }
        test_results.append(result)
        
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}; r_acc: {:.1f}; f_acc: {:.1f}".format(v_id, val, acc*100, ap*100, r_acc, f_acc))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
    # Lưu kết quả trung bình vào log
    log_data[testSet] = {
        'mean_acc': np.array(accs).mean(),
        'mean_ap': np.array(aps).mean(),
        'results': test_results
    }


    # Ghi log_data vào file JSON
    model_filename = os.path.basename(opt.model_path)  # model_vit_base_patch16_224_best_3.pth
    parent_folder = os.path.basename(os.path.dirname(opt.model_path))  # workspace
    output_filename = model_filename.replace('.pth', '_log.json')
    output_filename = parent_folder + '_' + output_filename
    with open(output_filename, 'w') as json_file:
        json.dump(log_data, json_file, indent=4)
    
    print("Log saved to ", output_filename)

