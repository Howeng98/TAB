from torch.utils.tensorboard import SummaryWriter

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import datetime
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from model import Main_Backbone, Projection_Head
from utils import save_config_file, count_parameters, save_grid_image, plot_figure
from test import test
from scripts.mvtec import MVTecDataset
from scripts.prompt_ensemble import encode_text_with_prompt_ensemble
from open_clip import *
from padim import padim_eval


def train(train_dataset, train_loader, args):
                        
    # Logging
    writer = SummaryWriter()
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # save config file
    save_config_file(writer.log_dir, args)

    # Model
    model = Main_Backbone()    
    image_projection_head = Projection_Head(in_channel=512)
    text_projection_head = Projection_Head(in_channel=512)
    
    model_clip, _, preprocess = create_model_and_transforms(args.model, args.input_size, pretrained=args.pretrained, jit=False)
    model_clip.to(args.device)
    tokenizer = get_tokenizer(args.model)

    # Parameters
    parameters = count_parameters(model) + count_parameters(image_projection_head) + count_parameters(text_projection_head)

    logging.info(f"Date: {datetime.datetime.now()}")
    logging.info(f"Note: {args.note}")
    logging.info(f"Parameters: {parameters:.3f}M parameters.")
    print("Model Parameters: {:3f}".format(parameters))
    
    # Feature Encoder Optimizer    
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        # list(filter(lambda p: p.requires_grad, image_projection_head.parameters())) +       
        # list(filter(lambda p: p.requires_grad, text_projection_head.parameters())),        
        lr=args.lr, betas=(0.5,0.999))

    # optimizer_text = torch.optim.AdamW(        
    #     list(filter(lambda p: p.requires_grad, text_projection_head.parameters())),
    #     lr=0.01, betas=(0.5,0.999))
    
    # optimizer_image = torch.optim.AdamW(        
    #     list(filter(lambda p: p.requires_grad, image_projection_head.parameters())),
    #     lr=0.01, betas=(0.5,0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)        
    
    # model
    model = model.to(args.device)
    # image_projection_head = image_projection_head.to(args.device)
    # text_projection_head = text_projection_head.to(args.device)
    
    logging.info(f"Backbone: {args.backbone}.")
    logging.info(f"Optimizer: {args.optm}.")
    logging.info(f"Synthetic Settings: {args.settings}.")
    logging.info(f"DataParallel: {args.data_parallel}.")
    logging.info(f"Epochs: {args.epochs} epochs.")
    logging.info(f"Batch_size: {args.batch_size}.")
    logging.info(f"Input_size: {args.input_size}.")
    logging.info(f"Training_Data: {len(train_loader)*args.batch_size}.")
    logging.info(f"Learning Rate: {args.lr}.")    
    logging.getLogger('matplotlib.font_manager').disabled = True
        
    criterion2 = torch.nn.MSELoss(reduction='mean')     
    
    # text prompts
    with torch.cuda.amp.autocast() and torch.no_grad():
        obj_list = train_dataset.__get_cls_names__() 
        # print(obj_list)
        normal_text_prompts, abnormal_text_prompts = encode_text_with_prompt_ensemble(model_clip, obj_list, tokenizer, args.device)    
        normal_text_prompts = normal_text_prompts.to(args.device)
        abnormal_text_prompts = abnormal_text_prompts.to(args.device)

        # normal_text_features = normal_text_features/normal_text_features.norm(dim=-1, keepdim=True)
        # anomaly_text_features = anomaly_text_features/anomaly_text_features.norm(dim=-1, keepdim=True)
    
    information_list = dict()
    information_list['total_loss'] = list()
    information_list['loss1'] = list()
    information_list['loss2'] = list()         
    information_list['loss3'] = list()        
    
    CLASS_NAMES = ['screw']
    train_loader_list = []
    test_loader_list = []

    # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

    for class_name in CLASS_NAMES:
        train_dataset = MVTecDataset('../MvTecAD', class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=4, pin_memory=False, num_workers=8)
        train_loader_list.append(train_dataloader)
        
        test_dataset = MVTecDataset('../MvTecAD', class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=8, pin_memory=True, num_workers=8)
        test_loader_list.append(test_dataloader)
        
        information_list[class_name+'_ImageAUROC'] = list()
        information_list[class_name+'_PixelAUROC'] = list()
        information_list[class_name+'_PixelPRO'] = list()        
            
    
    for epoch in range(args.epochs):
        
        # train
        model.train_mode()        
        tqdm_obj = tqdm(train_loader)                
        losses = []                        
                
        for x, z, class_names, class_label, mask in tqdm_obj:
            x = x.to(args.device)
            z = z.to(args.device)
            mask = mask.to(args.device)
            
            normal_class_label  = class_label.long().to(args.device)
            abnormal_class_label = torch.tensor(list(map(lambda i : i + args.num_classes, normal_class_label)), device=args.device)
            
            output_feat  = model(x)
            output_feat2 = model(z)
            
            loss1 = F.cross_entropy(output_feat['FS'][-1], normal_class_label, reduction='mean')
            loss2 = F.cross_entropy(output_feat2['FS'][-1], abnormal_class_label, reduction='mean')            
            loss = loss1 + loss2
            tqdm_obj.set_description("Epoch:{} | Total_loss:{:3f} |".format(epoch, loss))
            
            optimizer.zero_grad()                                                     
            loss.backward(retain_graph=True)
            losses.append(loss.item())                        
            optimizer.step()

        # scheduler.step()
    
        information_list['total_loss'].append(sum(losses)/len(losses))        
        
        # save model 
        checkpoint_name = 'checkpoint_{:04d}.pt'.format(epoch)
        checkpoint_path = os.path.join(writer.log_dir, checkpoint_name)        
        torch.save({
            'feature_extractor_state_dict': model.model['MS'].state_dict(),
        }, checkpoint_path)
        
        
        # test
        if epoch % 1 == 0:                        
            test_results = padim_eval(checkpoint_path, CLASS_NAMES, train_loader_list, test_loader_list)            
            for class_name in CLASS_NAMES:
                logging.debug("Epoch: {} | loss:{:.5f} | IMAGE_AUROC:{:.3f} | PIXEL_AUROC:{:.3f} | Class:{}".format(epoch, loss, test_results[class_name][0], test_results[class_name][1], class_name))
                information_list[class_name+'_ImageAUROC'].append(test_results[class_name][0]*100)
                information_list[class_name+'_PixelAUROC'].append(test_results[class_name][1]*100)
                plot_figure(epoch, information_list, type='test_result', class_names=CLASS_NAMES)
                
    logging.info("Training has finished.")
    logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")
