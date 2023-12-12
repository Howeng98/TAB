'''
==================================================================
This is a fine-tuned and test python file 
to run my anomaly backbone model 
to get Ours Detection and Localization results.
==================================================================
'''

import os
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from utils import Repeat, plot_fig, visualize_tsne, count_parameters
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from config import get_args
from pro_curve_util import compute_pro
from generic_util import trapezoid
from model import Main_Backbone, Discriminator


def save_results(class_name, auroc_sample, auroc_pixel, pixel_pro, loss, save_path):
    output_path = os.path.join(save_path, class_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    msg = ''
    msg += 'IMAGE_AUROC{:.3f} | PIXEL_AUROC:{:.3f} | PIXEL_PRO:{:.3f} | Loss:{:.3f} | Class:{}\n'.format(auroc_sample, auroc_pixel, pixel_pro, loss, class_name)

    with open(os.path.join(output_path, 'result.txt'), 'a+') as file:
        file.write(msg)

# fine-tuned
def fine_tuned_train(model, train_loader, test_loader, _class_, args, save_path):
    print("================== Training Phase ==================")       
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam([
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr":args.lr, "weight_decay":args.weight_decay},
        # {"params": filter(lambda p: p.requires_grad, model_disc.parameters()), "lr":0.1, "weight_decay":args.weight_decay},
    ])
    
    for epoch in range(args.epochs):
        
        # train        
        model.eval_mode()
        # model_disc.eval()
        tqdm_obj = tqdm(train_loader)
        loss = 0.0
        for x, y, mask in tqdm_obj:
            x = x.to(args.device)
            y = y.to(args.device).float() # all y is 0   
            mask = mask.to(args.device)
                                    
            output_feat = model(x)
            loss = model.cal_loss(feat_t=output_feat['FT'][:-1], feat_s=output_feat['FS'][:-1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            
            tqdm_obj.set_description("Epoch:{} | Loss:{:3f}".format(epoch, loss))
        
        # test
        model.eval_mode()
        # model_disc.eval()
        IMAGE_AUROC, PIXEL_AUROC, PIXEL_PRO = test(model, test_loader, _class_, args, save_path)
        print("IMAGE_AUROC:{:.3f} | PIXEL_AUROC:{:.3f} | PIXEL_PRO:{:.3f} |  Class:{}".format(IMAGE_AUROC, PIXEL_AUROC, PIXEL_PRO, _class_))
        save_results(_class_, IMAGE_AUROC, PIXEL_AUROC, PIXEL_PRO, loss, save_path)
    return model
    

def test(model, test_loader, _class_, args, save_path):    
    scores = []        
    gt_list = []
    gt_mask_list = []
    data_list = []
    
    tsne_embeds = []
    tsne_labels = []
    
    model.eval_mode()
    with torch.no_grad():
        for data, label, mask in tqdm(test_loader, disable=True):
            gt_list.extend(label.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            data_list.extend(data.cpu().detach().numpy())
            
            data = data.to(args.device)
            output_feat = model(data)
            score = model.cal_am(output_feat['FS'][:-1], output_feat['FT'][:-1])
            scores.extend(score)
            
            # use last two level feat to visualize
            tsne_embeds.extend(output_feat['FS'][-1].cpu().data.numpy())
            tsne_labels.extend(label)
                        
    tsne_embeds = np.asarray(tsne_embeds)
    tsne_labels = np.asarray(tsne_labels)
    # print(tsne_embeds.shape)
    tsne_embeds = tsne_embeds.reshape(tsne_embeds.shape[0], -1)
        
    tsne = TSNE(
            n_components=2, 
            verbose=0, 
            n_iter=20000,#10000
            learning_rate=350,
            perplexity=30,#5
            early_exaggeration=12,
            angle=0.5,
            init="pca",
        ).fit_transform(tsne_embeds)
    visualize_tsne(tsne, tsne_labels, _class_)
    
        
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list, dtype=int)    
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    # print("Image FPR:{} | TPR:{}".format(fpr, tpr)) 

    precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), img_scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    cls_threshold = thresholds[np.argmax(f1)]

    gt_mask = np.asarray(gt_mask_list, dtype=int)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]

    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    
    # print(gt_mask.squeeze().shape, scores.shape)
    all_fprs, all_pros = compute_pro(scores, gt_mask.squeeze(), 5000)
    au_pro = trapezoid(all_fprs, all_pros, x_max=0.3)
    au_pro /= 0.3
    
    # t_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
    save_dir = os.path.join(save_path, './'+_class_+'/')
    os.makedirs(save_dir, exist_ok=True)
    # plot_fig(data_list, scores, gt_mask_list, cls_threshold, save_dir, _class_, 10)
    
    return img_roc_auc, per_pixel_rocauc, au_pro



if __name__ == "__main__":
    
    CLASS_NAMES = [         
        'pill', 'capsule',
        'toothbrush', 'zipper', 'transistor', 'carpet', 'grid',        
        'bottle', 'cable', 'hazelnut', 'leather',
        'tile', 'wood', 
        'metal_nut',
        'screw', 
    ]
    resize_shape = 256
    crop_size = 224
    
    args = get_args()
    args.device = torch.device('cuda:0')
    args.epochs = 20
    args.batch_size = 32
    args.lr = 0.001 #0.001
    
    model = Main_Backbone()
    # ./runs/May20_17-35-38_cvlav-Z490-VISION-G/checkpoint_0099.pt
    # ./runs/May27_12-30-04_rn18/checkpoint_0070.pt (60.pt/127.pt)
    ckpt_dir = './runs/Aug08_14-52-36_cvlav-Z490-VISION-G/checkpoint_0025.pt'
    ckpt = torch.load(ckpt_dir, map_location=args.device)

    try:        
        model.model['MS'].load_state_dict(ckpt['feature_extractor_state_dict'], strict=False)
        print("Load ckpts from [feature_extractor_state_dict]")
    except:        
        model.model['MS'].load_state_dict(ckpt['backbone_state_dict'], strict=False)
        print("Load ckpts from [backbone]")
        
    print("Param:{}".format(count_parameters(model)))
    model_disc = Discriminator(in_channel=256)
    
    model = model.to(args.device)
    model_disc = model_disc.to(args.device)
    
    avg_image_auroc = 0
    avg_pixel_auroc = 0
    avg_pixel_pro = 0
    
    for class_name in CLASS_NAMES:
        print("Class_name: {}".format(class_name))
        train_dataset = MVTecDataset('../MvTecAD', class_name=class_name, is_train=True, resize=resize_shape, cropsize=crop_size)
        test_dataset = MVTecDataset('../MvTecAD', class_name=class_name, is_train=False, resize=resize_shape, cropsize=crop_size)
        
        train_loader = torch.utils.data.DataLoader(Repeat(train_dataset, 5), batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=5)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    
        # FINE-TUNED
        model = fine_tuned_train(model, train_loader, test_loader, class_name, args, './results_all')
    
        # TESTING
        IMAGE_AUROC, PIXEL_AUROC, PIXEL_PRO = test(model, test_loader, class_name, args, './results_all')
        
        avg_image_auroc += IMAGE_AUROC
        avg_pixel_auroc += PIXEL_AUROC
        avg_pixel_pro   += PIXEL_PRO
        print("IMAGE_AUROC:{:.3f} | PIXEL_AUROC:{:.3f} | PIXEL_PRO:{:.3f} | Class:{}".format(IMAGE_AUROC, PIXEL_AUROC, PIXEL_PRO, class_name))
    
    avg_image_auroc /= len(CLASS_NAMES)
    avg_pixel_auroc /= len(CLASS_NAMES)
    avg_pixel_pro /= len(CLASS_NAMES)
    print("=============================================\nAVG_IMAGE_AUROC:{:.3f} | AVG_PIXEL_AUROC:{:.3f} | AVG_PIXEL_PRO:{:.3f}\n=============================================".format(avg_image_auroc, avg_pixel_auroc, avg_pixel_pro))