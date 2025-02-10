import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.model import build_model
from networks.base_model import BaseModel, init_weights
import torch.nn.functional as F


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        
        self.features_teacher_512 = None
        self.features_student_512 = None
        self.alpha = 0.5
        self.T = 10
        self.DKL = nn.KLDivLoss()
        self.criterion_ce = nn.CrossEntropyLoss()   
        self.d_loss = 0.0
        
        
        
        if self.isTrain and not opt.continue_train:
            #self.model = resnet50(pretrained=False, num_classes=1)
            self.model = build_model(backbone=opt.backbone, num_features=opt.num_features, pretrained=False, num_classes=1, freeze_exclude=None)
            self.model_teacher = resnet50(pretrained=False, num_classes=1)
            self.model_teacher.load_state_dict(torch.load("./weights/ADOF_model_epoch_9.pth", map_location='cpu'), strict=True)
            self.model_teacher.eval()


        self.student_handle = self.model.backbone.avgpool.register_forward_hook(self.student_hook)
        self.teacher_handle = self.model_teacher.avgpool.register_forward_hook(self.teacher_hook)


        if not self.isTrain or opt.continue_train:
            self.model = build_model(backbone=opt.backbone, num_features=opt.num_features, pretrained=False, num_classes=1, freeze_exclude=None)

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
            self.model_teacher.to(opt.gpu_ids[0])
 
    def teacher_hook(self, module, input, output):
        self.features_teacher_512 = output.view(output.size(0), -1)  # Flatten
        
    def student_hook(self, module, input, output):
        """ Hook để lấy feature 512-dim của Student """
        self.features_student_512 = output.view(output.size(0), -1)  # Flatten
        
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
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        
        with torch.no_grad():
           self.output_teacher = self.model_teacher(self.input)
            
        self.output = self.model(self.input)
        
    def distillation_loss(self):
        """ Tính loss KL-Divergence giữa Student và Teacher """
        
        # loss = nn.KLDivLoss(reduction="batchmean")(  
        #     torch.log_softmax(student_logits / self.T, dim=1),
        #     torch.softmax(teacher_logits / self.T, dim=1),
        # )
        
        loss = self.criterion_ce(self.features_student_512, self.features_teacher_512)
        
        return loss
        
    def get_loss(self):

        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.d_loss = self.distillation_loss()
        self.loss = (1-self.alpha)*self.loss_fn(self.output.squeeze(1), self.label) + self.alpha * self.d_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    

if __name__ == '__main__':
    model_teacher = resnet50(num_classes=1)
    model_teacher.load_state_dict(torch.load("../weights/ADOF_model_epoch_9.pth", map_location='cpu'), strict=True)








