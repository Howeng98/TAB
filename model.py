import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resnet import resnet18, resnet50, wide_resnet50_2

class Main_Backbone(nn.Module):
    def __init__(self, backbone_name, text_input_dim, visual_input_dim, out_dim):
        super(Main_Backbone, self).__init__()
        
        self.backbone_name = backbone_name
        self.init_model()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  
        
        self.normal_text_projection_head  = Projection_Head(in_channel=text_input_dim, out_channel=out_dim)        
        self.abnormal_text_projection_head  = Projection_Head(in_channel=text_input_dim, out_channel=out_dim)
          
        self.normal_image_projection_head = Projection_Head(in_channel=visual_input_dim, out_channel=out_dim)
        self.abnormal_image_projection_head = Projection_Head(in_channel=visual_input_dim, out_channel=out_dim)        
        
        # self.cnn_fc = nn.Linear(512, 512)
        # self.clip_fc = nn.Linear(512, 512)
        # self.relu = nn.ReLU()
        # self.pool = torch.nn.AvgPool2d(2, stride=2)
                
        
    def init_model(self):              
        
        if self.backbone_name == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif self.backbone_name == 'wide_resnet50_2':
            self.backbone = wide_resnet50_2(pretrained=True)
        elif self.backbone_name == 'resnet50':
            self.backbone = resnet50(pretrained=True)
        else:
            raise RuntimeError(f'{self.backbone_name} is not found.')
        
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, z):
        self.normal_CNN_visual_feat   = self.backbone(x)[-1]
        self.normal_CNN_visual_feat   = self.normal_image_projection_head(self.normal_CNN_visual_feat)
        self.abnormal_CNN_visual_feat = self.backbone(z)[-1]
        self.abnormal_CNN_visual_feat   = self.abnormal_image_projection_head(self.abnormal_CNN_visual_feat)
        
        self.normal_CNN_visual_feat = F.normalize(self.normal_CNN_visual_feat, p=2)
        self.abnormal_CNN_visual_feat = F.normalize(self.abnormal_CNN_visual_feat, p=2)
                        
        return self.normal_CNN_visual_feat, self.abnormal_CNN_visual_feat
        
    def train_mode(self):        
        self.backbone.train()
        self.normal_text_projection_head.train()
        self.abnormal_text_projection_head.train()
        self.normal_image_projection_head.train()
        self.abnormal_image_projection_head.train()
            
    def eval_mode(self):        
        self.backbone.eval()
        self.normal_text_projection_head.eval()        
        self.abnormal_text_projection_head.eval()        
        self.normal_image_projection_head.eval()
        self.abnormal_image_projection_head.eval()
        
    def reproject(self, cnn_features, clip_features):        
        cnn_features  = self.relu(self.cnn_fc(cnn_features))
        clip_features = self.relu(self.clip_fc(clip_features))        
        mixed_features = cnn_features + clip_features
        mixed_features = F.normalize(mixed_features, p=2)        
        return mixed_features
        
    def cal_loss(self, z1, z2, labels):
        # print(z1.shape, z2.shape)
        logits = self.cal_logits(z1, z2)        
        return F.cross_entropy(logits, labels)
    
    def cal_logits(self, z1, z2):
        return torch.inner(z1, z2) * self.logit_scale
    
    def save_checkpoint(self, writer, epoch):
        checkpoint_name = 'checkpoint_{:04d}.pt'.format(epoch)  
        checkpoint_path = os.path.join(writer.log_dir, checkpoint_name)    
        torch.save(self.backbone.state_dict(), checkpoint_path)
        return checkpoint_path
        
                
# reference from https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2
class Projection_Head(nn.Module):
    def __init__(self, in_channel=512, out_channel=512):
        super(Projection_Head, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel)
        self.fc2 = nn.Linear(out_channel, out_channel)        
        self.bn1 = nn.BatchNorm1d(out_channel)        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(out_channel)        
        # self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        projected = self.fc(x)
        x = self.relu(projected)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x