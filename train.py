import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
import json
####################################################
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
#####################################################

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


# test config 
#Training folder struct (The name of the folder can be changed)
#ForenSynths_train_val/
#│
#├── train/
#│   └── [...]
#│
#├── val/
#│   └── [...]
#│
#└── test/
#    └── [...]

vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]
#vals = ['car', 'cat', 'chair', 'horse']
#multiclass = [0, 0, 0, 0]

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


from functools import wraps

# Decorator to log metrics if USE_COMET is True
def log_comet_metric():
    def decorator(func):
        @wraps(func)
        def wrapper(metric_name, *args, **kwargs):
            value, step, epoch = func(*args, **kwargs)  # Execute the original function
            if experiment is not None:
                # Assuming total_steps and experiment are globally available
                experiment.log_metric(metric_name, value, step=step, epoch = epoch)
            return value, step
        return wrapper
    return decorator

@log_comet_metric()
def log_metric(value, step=None, epoch=None):
    return value, step, epoch

if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    #############################################################
    experiment = None
    USE_COMET = opt.use_comet
    if USE_COMET:
        experiment = Experiment(
          api_key="MS89D8M6skI3vIQQvamYwDgEc",
          project_name="adof",
          workspace="danhvohoai2-gmail-com"
        )
    #############################################################
    
    seed_torch(100)
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    
    if experiment is not None:
        opt_dict = vars(opt)
        experiment.log_parameters(opt_dict)
    
    def testmodel(step=None, epoch=None):
        global experiment
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    
        all_results = []
    
        for v_id, val in enumerate(vals):
            Testopt.dataroot = f'{Testdataroot}/{val}'
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = False
            Testopt.no_crop = False
    
            results = validate(model.model, Testopt, return_images=False)
    
            print(f"({v_id} {val:12}) acc: {results['acc']*100:.4f}; ap: {results['ap']*100:.4f}; "
                  f"r_acc: {results['r_acc']:.4f}; f_acc: {results['f_acc']:.4f}")
    
            # Save per-class result
            result_to_save = results.copy()
            result_to_save['dataset'] = val
            if isinstance(result_to_save['confusion_matrix'], np.ndarray):
                result_to_save['confusion_matrix'] = result_to_save['confusion_matrix'].tolist()
            all_results.append(result_to_save)
    
            # Log to Comet
            if experiment is not None:
                for k, v in results.items():
                    if isinstance(v, (float, int)):
                        experiment.log_metric(f"test/{val}/{k}", v, step=step, epoch=epoch)
    
        # Compute mean over all classes
        mean_metrics = {}
        keys = ['acc', 'ap', 'r_acc', 'f_acc', 'precision', 'recall', 'f1', 'auc']
        for key in keys:
            values = [res[key] for res in all_results if key in res]
            mean_metrics[key] = np.mean(values)
    
        print(f"(Mean         ) acc: {mean_metrics['acc']*100:.4f}; ap: {mean_metrics['ap']*100:.4f}")
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    
        # Save all results to JSON
        with open(f'results_epoch_{epoch or "final"}.json', 'w') as f:
            json.dump(all_results, f, indent=4)
    
        # Log mean metrics to Comet
        if experiment is not None:
            for k, v in mean_metrics.items():
                experiment.log_metric(f"test/mean_{k}", v, step=step, epoch=epoch)
    
        return all_results    

    # def testmodel(step=None, epoch=None):
    #     global experiment  # Declare that we are using the global 'experiment'
    #     print('*'*25);accs = [];aps = []
    #     print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    #     for v_id, val in enumerate(vals):
    #         Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
    #         Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
    #         Testopt.no_resize = False
    #         Testopt.no_crop = False
    #         acc, ap, r_acc, f_acc, _, _ = validate(model.model, Testopt)
    #         accs.append(acc);aps.append(ap)
    #         print("({} {:12}) acc: {:.4f}; ap: {:.4f}; r_acc: {:.4f}; f_acc: {:.4f}".format(v_id, val, acc*100, ap*100, r_acc, f_acc))
        
    #         # Log the metrics for Comet
    #         #experiment.log_metric(f"test/acc_{val}", acc * 100)
    #         #experiment.log_metric(f"test/ap_{val}", ap * 100)
    #         #experiment.log_metric(f"test/r_acc_{val}", r_acc)
    #         #experiment.log_metric(f"test/f_acc_{val}", f_acc)
        
    #     print("({} {:10}) acc: {:.4f}; ap: {:.4f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
    #     print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
     
    #     # Log the mean values
    #     mean_acc = np.array(accs).mean() * 100
    #     mean_ap = np.array(aps).mean() * 100

    #     log_metric("test/acc", mean_acc, step=step, epoch=epoch)
    #     log_metric("test/ap", mean_ap, step=step, epoch=epoch)

        
    # Run for the first time to test the code for any errors
    model.eval();testmodel();

    model.train()
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):
        start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                log_metric("train/loss", model.loss, step=model.total_steps, epoch = epoch)  # Log the current loss directly


        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            

        # Validation
        model.save_networks(f'{epoch}')

            
        print(' Epoch', epoch,':',start_time,"-->", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        model.save_networks('last')
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)

        log_metric("validation/loss", model.loss, step=model.total_steps, epoch = epoch)  # Log the current loss directly
        log_metric("validation/acc", acc, step=model.total_steps, epoch = epoch)  # Log the current loss directly
        log_metric("validation/ap", ap, step=model.total_steps, epoch = epoch)  # Log the current loss directly

        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        testmodel(step = model.total_steps, epoch = epoch)
        model.train()

    #model.eval();testmodel()
    model.save_networks('last')
    
