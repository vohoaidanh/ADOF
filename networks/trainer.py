import functools
import torch
import torch.nn as nn
#from networks.resnet import resnet50
from networks.model import build_model
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            #self.model = resnet50(pretrained=False, num_classes=1)
            self.model = build_model(backbone=opt.backbone, num_features=opt.num_features, pretrained=True, num_classes=1, freeze_exclude=None)


        if not self.isTrain or opt.continue_train:
            self.model = build_model(backbone=opt.backbone, num_features=opt.num_features, pretrained=True, num_classes=1, freeze_exclude=None)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        
        if torch.cuda.is_available() and len(opt.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model)  # Sử dụng DataParallel
            
        if torch.cuda.is_available():
            self.model.to(opt.gpu_ids[0])
 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        if isinstance(input, tuple):
            self.input = (input[0][0].to(self.device), input[0][1].to(self.device))
        else:
            self.input = input[0].to(self.device)
        
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)


    def adjust_gradient(self):
        label_condition = (self.label == 1)

        if label_condition.any():  # Nếu có mẫu nào có nhãn là 1
            for name, param in self.model.reference.named_parameters():
                if param.grad is not None:
                    param.grad *= torch.where(label_condition.unsqueeze(1), 2.0, 1.0).type_as(param.grad)
        
    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.adjust_gradient()
        self.optimizer.step()
        
        
        
        
        
        

