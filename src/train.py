from torch.utils.tensorboard import SummaryWriter

import os
import torch
import torch.nn.functional as F
import numpy as np
import datetime
import logging

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Main_Backbone, Projection_Head
from utils import save_config_file, count_parameters, plot_figure, setup_seed, save_grid_image
from mvtec import MVTecDataset
from prompt_ensemble import encode_text_with_prompt_ensemble
from open_clip import *
from padim import padim_eval
from scripts.loss import FocalLoss

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def train(train_dataset, train_loader, args):
                        
    # Logging
    writer = SummaryWriter()
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # save config file
    save_config_file(writer.log_dir, args)
    
    # Model
    model = Main_Backbone(args.backbone_name, text_input_dim=512, visual_input_dim=2048, out_dim=512)
    
    
    model_clip, _, preprocess = create_model_and_transforms(args.model, args.input_size, pretrained=args.pretrained, jit=False)
    model_clip = model_clip.to(args.device)
    tokenizer = get_tokenizer(args.model)

    # Parameters
    parameters = count_parameters(model) 
    print("Model Parameters: {:3f}".format(parameters))
        
    # Feature Encoder Optimizer    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)    

    # model
    model = model.to(args.device)

    logging.info(f"Date: {datetime.datetime.now()}")
    logging.info(f"Note: {args.note}")
    logging.info(f"Parameters: {parameters:.3f}M parameters.")
    logging.info(f"Backbone: {args.backbone_name}.")
    logging.info(f"Optimizer: {args.optm}.")
    logging.info(f"Synthetic Settings: {args.settings}.")    
    logging.info(f"Epochs: {args.epochs} epochs.")
    logging.info(f"Batch_size: {args.batch_size}.")
    logging.info(f"Input_size: {args.input_size}.")
    logging.info(f"Training_Data: {len(train_loader)*args.batch_size}.")
    logging.info(f"Learning Rate: {args.lr}.")    
    logging.getLogger('matplotlib.font_manager').disabled = True
        
    
    # text prompts
    with torch.cuda.amp.autocast() and torch.no_grad():
        obj_list = train_dataset.__get_cls_names__() 
        
        normal_text_prompts, abnormal_text_prompts = encode_text_with_prompt_ensemble( model_clip, obj_list, tokenizer, args.device)    
        normal_CLIP_text_feat = normal_text_prompts.to(args.device)
        abnormal_CLIP_text_feat = abnormal_text_prompts.to(args.device)
        
        CLIP_text_feat = torch.cat((normal_CLIP_text_feat, abnormal_CLIP_text_feat), dim=0)

        # normal_text_features = normal_text_prompts/normal_text_prompts.norm(dim=-1, keepdim=True)
        # anomaly_text_features = abnormal_text_prompts/abnormal_text_prompts.norm(dim=-1, keepdim=True)
    
        
    CLASS_NAMES = ['grid']
    train_loader_list = []
    test_loader_list = []
    
    for class_name in CLASS_NAMES:
        padim_train_dataset = MVTecDataset('../MvTecAD', class_name=class_name, is_train=True)
        train_dataloader = DataLoader(padim_train_dataset, batch_size=8, pin_memory=True, num_workers=8)
        train_loader_list.append(train_dataloader)
        
        padim_test_dataset = MVTecDataset('../MvTecAD', class_name=class_name, is_train=False)
        test_dataloader = DataLoader(padim_test_dataset, batch_size=8, pin_memory=True, num_workers=8)
        test_loader_list.append(test_dataloader)                
    
   
    
    for epoch in range(args.epochs):        
        # train
        model.train_mode()
        tqdm_obj = tqdm(train_loader)                
        losses = []                        
                
        for x, mask, class_names, class_labels in tqdm_obj:
            x = x.to(args.device)            
            mask = mask.to(args.device)
            class_labels = class_labels.long().to(args.device)
            
            CNN_visual_feat = model(x)           
            
            # clip visual encoder            
            with torch.no_grad():
                clip_visual_feat, patch_tokens = model_clip.encode_image(x, args.features_list)
                clip_visual_feat = clip_visual_feat / clip_visual_feat.norm(dim=-1, keepdim=True)
            # normal_CLIP_visual_feat = image_projection_head(clip_x)
            # normal_CLIP_visual_feat = F.normalize(clip_x, p=2)
            # abnormal_CLIP_visual_feat = image_projection_head_1(clip_z)
            # abnormal_CLIP_visual_feat = F.normalize(clip_z, p=2)            
            
            # text re-projection and normalized            
            # normal_CLIP_text_feat  = text_projection_head(normal_CLIP_text_feat)            
            # normal_CLIP_text_feat = F.normalize(normal_CLIP_text_feat, p=2)                        
            # abnormal_CLIP_text_feat = text_projection_head_1(abnormal_CLIP_text_feat)
            # abnormal_CLIP_text_feat = F.normalize(abnormal_CLIP_text_feat, p=2)
      
            mixed_feat = model.reproject(cnn_features=CNN_visual_feat, patch_tokens=clip_visual_feat)
            loss = model.cal_loss(mixed_feat, CLIP_text_feat, class_labels)
            # loss2 = model.cal_loss(abnormal_CNN_visual_feat, abnormal_CLIP_text_feat, class_label)
            # loss = (loss1 + loss2)/2

            tqdm_obj.set_description("Epoch:{} | Total_loss:{:3f}".format(epoch, loss))            
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()          
                                   
        scheduler1.step()
        
        # save model 
        checkpoint_path = model.save_checkpoint(writer, epoch)
                
        # test
        if epoch%1 == 0:            
            model.eval_mode()
            test_results = padim_eval(checkpoint_path, CLASS_NAMES, train_loader_list, test_loader_list)    
            print('========================================================') 
            for class_name in CLASS_NAMES:
                logging.debug("Epoch: {} | loss:{:.5f} | IMAGE_AUROC:{:.3f} | PIXEL_AUROC:{:.3f} | Class:{}".format(epoch, loss, test_results[class_name][0], test_results[class_name][1], class_name))                
                print(' Image-AUROC: {:.3f} | Pixel-AUROC: {:.3f} | Class-Name:{}'.format(test_results[class_name][0], test_results[class_name][1], class_name))
            print(' Average-Image-AUROC: {:.3f} | Average-Pixel-AUROC: {:.3f}'.format(test_results['avg_image_auroc'], test_results['avg_pixel_auroc']))
            print('========================================================')
