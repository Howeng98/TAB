# Anomaly Backbone Survey Log
![Status](https://img.shields.io/static/v1.svg?label=Status&message=progressing&color=green)

### Pretrained backbone TODO list :heavy_check_mark: 
Idx | Content | State | Note
:------------ | :-------------| :-------------| :-------------
1 | Try model in supervised with MvTecAD | :heavy_check_mark: | 
2 | Try model in unsupervised with MvTecAD | :heavy_check_mark: | 
3 | Find a model fit with high dimensions input | |
4 | Classify dataset into several classes | :heavy_check_mark: | One folder with multiple classes
5 | Make a dataloader | :heavy_check_mark: |
6 | Fine-tuned resnet FC from 1000 to CLASS_NUM | :heavy_check_mark: |
7 | Check evaluate model PaDiM whether have fine-tuned or not | :heavy_check_mark: | PaDiM no fine-tuned
8 | Add more classes into traning dataset to show that more data more useful | :heavy_check_mark: | It show improve but no that significant(+1.0%)
9 | Contrastive Learning model | :heavy_check_mark: | 
10 | Setup positive and negative pair | :heavy_check_mark: | 


## 整個流程會分成三個部分。

### 第一部分： 
Pre-Training | 用SimCLR的CL架構去訓練manufacturing(或者ImageNet）資料集以得到一個 pretrained 的 init weights。

### 第二部分：
Fine-Tune | 這裡使用第一部分訓練出來的 pretrained 權重（這部分的resnet encoder 權重是freezed的），這裡要使用CIFAR10/STL10的 training dataset 來 fine-tuned linear classifier。(由於Encoder的權重都是freeze的，所以其實使用SimCLR的孿生網路架構還是只是用Resnet一個encoder都是一樣的意思。)

### 第三部分：
Testing | 這裡就是 evaluate 的階段，不管是encoder還是classifier的權重全部都是freeze的，不更新。跑在 testing dataset 上。



## Results on **STL10** / ImageNet Pretrained (For reference)
| Base Net  | Project Head | Feature  | Optimizer | Learning Rate | Weight Decay | Epochs | Top 1 Accuracy | Top 5 Accuracy |
| :-------: | :----------: | :-----:  | :-------: | :-----------: | :----------: | :----: | :------------: |  :------------: | 
| ResNet-18(ref) |     128      |   512    |   Adam   |   1e-3  |     0.3      |  100   |     78.94%     |  |
| ResNet-34(ref) |     128      |   512    |   Adam   |   1e-3   |     0.3      |  100   |     79.34%     |   |

---

## Results on **STL10** / Manufacturing Pretrained (Ours)
| Base Net  | Project Head | Feature  | Optimizer | Learning Rate | Weight Decay | Epochs | Top 1 Accuracy | Top 5 Accuracy |
| :-------: | :----------: | :-----:  | :-------: | :-----------: | :----------: | :----: | :------------: |  :------------: | 
| ResNet-18(pretrained) |     128      |   512   |   Adam   |   1e-3   |     0.3     |  100   |     78.06%     |  98.51% |
| ResNet-18(ours) |     128      |   512   |   Adam   |   1e-2   |     0.001     |  100   |     50%     |  91.27% |
| ResNet-18(ours) |     128      |   512   |   Adam   |   1e-2   |     0.001     |  100   |    38.90%     |  88.28% |

## Results on **CIFAR10** / Manufacturing Pretrained (Ours)
| Base Net  | Project Head | Feature  | Optimizer | Learning Rate | Weight Decay | Epochs | Top 1 Accuracy | Top 5 Accuracy |
| :-------: | :----------: | :-----:  | :-------: | :-----------: | :----------: | :----: | :------------: |  :------------: | 
| ResNet-18(pretrained) |     128      |   512   |   Adam   |   1e-3   |     0.3     |  100   |     44.42%     |  89.50% |
| ResNet-18(ours) |     128      |   512   |   Adam   |   1e-2   |     0.001     |  100   |     39.33%     |  86.58% |


### Explanations Video
[Recurrent and Residual U-Net](https://www.youtube.com/watch?v=7aDOtKN2cJs)
[Attention Unet](https://www.youtube.com/watch?v=KOF38xAvo8I)



06/15
- Make a dataloder to read all data (7classes) from the dataset folder
- Change the Resnet18 FC in_features linear from 1000 to CLASS_NUM 
- Check PaDiM got fine-tune or not

06/23
- Training on DGAM + AITEX
- 17 classes (7185 samples)
- Tuning on CFlow-AD
- Evaluate on MvTecAD
- Training dataset lack of variant
- Find more data to enrich dataset samples

06/26
- Add some utils and tools script (time.py, script)
- Add classes number from 17 to 25
- Remove unbalanced classes(e.g. AITEX) , reduce classes number from 25 to 18
- Retrain backbone with weight_decay and validation set
- ResNet18 Model Parameters: 11,185,746

06/29
- Try Unet-ResNet18
- UNet-ResNet18 Model Paramaters: 28,977,426
- Overall avarage AUROC: 79.70

07/07
- CBAM resnet Model Paramaters: 11,357,010

07/15
- Find a proper contrastive learning framework

07/20
- Update SimCLR framework and give a highest AUROC performance ever
- AVERAGE AUROC: 80.37 (SimCLR + 18classes + 100epcohs)
- More backbone training steps brings benefits to model, but not significantly (80.37->81.09 in 500 epochs)

    TODO
    - Going to setup my pretext tasks (Positive and Negative pairs
    - Enrich dataset :heavy_check_mark:
    - Use a better powerful encoder :heavy_check_mark:
    - Calculate detection from segmentation anomaly map :heavy_check_mark:

07/29
- add Triplet Loss

08/15
- add accumulate gradient update every 2 epochs
- temperature argument is very important, 0.5 only make loss decrease into 6.9, but 0.1 can make loss decrease into 4.50.

08/18
- add linear classifier as evaluation protocol
- modify self.backbone.fc with identity()
- add model parameters() and save checkpoint every n steps
- current params (resnet18 + projection_head): 11.506M
- train 1000 epochs, reach 3.05 loss
- but STL10 top1_acc only 40% (ImageNet pretrained: 78%)

08/19
- add input_size from 32 to 64 can make loss decrease to 1.5
- greater input_size make loss decrease more

10/10
- update current version beofre modifying model structure
- Fix load_state_dict() problem, now our evaluate linear regression(classification) and RD4AD model can load weights correctly.
- Save model parameters with model.backbone.state_dict() and model.projection_head.state_dict() / and model.module.backbone. .... for DataParallel model
- Make dataloader generate original_image, transformed_image and defect_image(with random sample from dataset) return 3 outputs.
- Add Triplet Loss
- STL10 evaluate.py: Top1: 38.90% | Top5: 88.28%
- Next target structure
<p align="center">
    <img src="https://user-images.githubusercontent.com/44123278/194815742-295388f8-31f8-40db-bfe7-c199a16d85e0.png">
</p>

10/19
- Add NSA synthetic image.
- But the quality of synthetic image is not clear with texture and object classes. (Gonna fix)

10/20
- Fix the quality of synthetic image is not clear problem!
- But the performance with NSA as defect image in this model structure is terrible.
- Next target: Change loss, clean dataset, realistic synthetic image, change model

11/04
- Siamese Network will cause collapsing problem when optimized without negative pairs.

11/15
- Add SSL framework (predict relative position & direction)
- Future work keyword: METRIC LEARNING

12/15
- add PsCo

01/06
- add new loss function

02/20
- add SPD loss
- focus on alignment information (such as remove vertical and horizontal flip in augmentation to remain the alignment information)


03/05
- add TripletMarginLoss

03/09
- Update current version 
- Start trying CDO and t-s structure

03/10
- RD4AD need to fix and let encoder.eval() at all time for better result, because it's teacher-student network
- Save current best version for further debug

03/14
- implement CDO as a new framework structure (teacher-student), MOM, OOM
- Detection performance reach 97.52 in RD4AD(paper: 98.46)
- Only got 66% in Detection in RegAD(8shots) | Paper performance is 90% (this show the CDO model still have some problems and space to improve)
- But still have some classes are not well, maybe can try some method in training phase to make representation can handle these classes

03/27
- add resize 320 + centercrop 224 (very important, can remove edge effects)
- try NSA / Cuspaste(NSA version), but seem like both performance same same
- using more dataset provide a little bit improve in perfomance, but it is a good work
- archieve 90.31% detection and 96.71% in localization in RegAD 8-shots
- in RegAD fine-tuning phase, I found that our model make a competitive (almost 90%) accuracy in the after first test phase in detection (zero-shot, RegAD run test instead of train phase first)

03/29
- Use gamme=3 better than gamme=2 (but not very efficient)
- Teacher and student use different structure doesn't bring benefit(I use Resnet34-Resnet18), loss only reach 6, or 7. But same structure can go deeper
