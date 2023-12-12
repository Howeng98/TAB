import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from resnet import resnet18, wide_resnet50_2
from scipy.ndimage import gaussian_filter
from weights_init import trunc_normal_
from scripts.loss import FocalLoss

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Main_Backbone(nn.Module):
    def __init__(self, backbone_name, text_input_dim, visual_input_dim, out_dim):
        super(Main_Backbone, self).__init__()
        
        self.backbone_name = backbone_name
        self.init_model()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  
        self.criterion = FocalLoss(class_num=30, size_average=True, gamma=2)
        
        self.normal_text_projection_head  = Projection_Head(in_channel=text_input_dim, out_channel=out_dim)        
        self.abnormal_text_projection_head  = Projection_Head(in_channel=text_input_dim, out_channel=out_dim)        
        self.normal_image_projection_head = Projection_Head(in_channel=visual_input_dim, out_channel=out_dim)
        self.abnormal_image_projection_head = Projection_Head(in_channel=visual_input_dim, out_channel=out_dim)        
        # self.normal_text_projection_head.apply(init_weights)
        # self.abnormal_text_projection_head.apply(init_weights)
        
        self.cnn_fc = nn.Linear(2048, 512)
        self.clip_fc = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        
        
    def init_model(self):              
        
        if self.backbone_name == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif self.backbone_name == 'wide_resnet50_2':
            self.backbone = wide_resnet50_2(pretrained=True)
        else:
            raise RuntimeError(f'{self.backbone_name} is not found.')
        
        # in_channel = student.fc.in_features
        # student.fc = nn.Linear(in_channel, 60)
        for param in self.backbone.parameters():
            param.requires_grad = True
        # for param in self.backbone.layer4.parameters():
        #     param.requires_grad = True
        
    def forward(self, x):
        self.normal_CNN_visual_feat   = self.backbone(x)[-1]
        # self.abnormal_CNN_visual_feat = self.backbone(z)[-1]
        self.normal_CNN_visual_feat   = F.normalize(self.normal_CNN_visual_feat, p=2)
        # self.abnormal_CNN_visual_feat = F.normalize(self.abnormal_CNN_visual_feat, p=2)
                        
        return self.normal_CNN_visual_feat
        
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
        
    def reproject(self, cnn_features, patch_tokens):        
        cnn_features  = self.relu(self.cnn_fc(cnn_features))
        clip_features = self.relu(self.clip_fc(patch_tokens))
        # print(cnn_features.shape, clip_features.shape)
        mixed_features = cnn_features + clip_features
        mixed_features = F.normalize(mixed_features, p=2)
        
        return mixed_features
        
    def cal_logits(self, z1, z2):
        # z1 = self.normal_image_projection_head(z1)
        # z2 = self.normal_text_projection_head(z2)
        
        z1 = F.normalize(z1, p=2)
        z2 = F.normalize(z2, p=2)
        return (z1 @ z2.T) * self.logit_scale
        
    def cal_loss(self, z1, z2, labels):
        logits = self.cal_logits(z1, z2)        
        return self.criterion(logits, labels)
    
    def save_checkpoint(self, writer, epoch):
        checkpoint_name = 'checkpoint_{:04d}.pt'.format(epoch)  
        checkpoint_path = os.path.join(writer.log_dir, checkpoint_name)    
        torch.save(self.backbone.state_dict(), checkpoint_path)
        return checkpoint_path
        
                
# copy from https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2
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

class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()        
        
        self.fc0 = nn.Linear(in_channel, 2)
        self.bn = nn.BatchNorm1d(2)        
        
        self.fc1 = nn.Linear(128, 2)
        self.bn1 = nn.BatchNorm1d(2)
        
        self.relu = nn.ReLU()
        # self.leakyrelu1 = nn.LeakyReLU(0.2)
        
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        # self.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.bn(x)
        x = self.relu(x)
                
        # x = self.dropout(x)
        # x = self.fc1(x)
        # x = self.bn1(x)
        x = self.softmax(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, out_channels)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    
# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x
        
        
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        if 'ViT' in model:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens