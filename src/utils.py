import os
import shutil
import torch
import yaml
import h5py
import math
import random
import cv2
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import MDS
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.ticker as mtick
from glob import glob
from torch import optim
from tqdm import tqdm
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torchvision.utils import save_image, make_grid
from torch.optim.optimizer import Optimizer
from bisect import bisect
from scipy.ndimage.measurements import label
from torch.utils.data import Dataset
import warnings
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from skimage import morphology
from skimage.segmentation import mark_boundaries
from typing import Any, Dict, Tuple, Union
import matplotlib.ticker as mticker

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

pos_to_diff = {
    0: (1, 0),
    1: (math.sqrt(3)/2, 1/2),
    2: (1/2, math.sqrt(3)/2),
    3: (0, 1),
    4: ((-1)*(1/2), math.sqrt(3)/2),
    5: ((-1)*(math.sqrt(3)/2), 1/2),
    6: (-1, 0),
    7: ((-1)*(math.sqrt(3)/2), (-1)*(1/2)),
    8: ((-1)*(1/2), (-1)*(math.sqrt(3)/2)),
    9: (0, -1),
    10: (1/2, (-1)*(math.sqrt(3)/2)),
    11: (math.sqrt(3)/2, (-1)*1/2)
}

def plot_figure(epoch, information_list, type='losses', class_names=None):
    if type == 'losses':
        plt.figure()
        plt.plot(information_list['total_loss'], color='green', label='total_loss')
        # plt.plot(information_list['loss1'], color='red', label='loss1')
        # plt.plot(information_list['loss2'], color='blue', label='loss2')
        # plt.plot(information_list['loss3'], color='green', label='loss3')
        plt.xlabel('epochs')
        plt.ylabel('total loss')
        plt.title('All losses')
        plt.legend()
        plt.savefig('./imgs/total_loss.png')
    else: #performance
        for class_name in class_names:
            plt.figure()
            plt.plot(information_list[class_name+'_ImageAUROC'], color='red', label='ImageAUROC')
            plt.plot(information_list[class_name+'_PixelAUROC'], color='green', label='PixelAUROC')
            # plt.plot(information_list[class_name+'_PixelPRO'], color='purple', label='PixelPRO')
            plt.xlabel('epochs')
            plt.ylabel('scores')
            plt.title(class_name+' AUROC and PRO scores')
            plt.legend()
            plt.savefig('./imgs/'+class_name+'_auroc_pro.png')


def set_class_label(PATH, CLASS_NAMES: dict, counter=0):
    class_names = os.listdir(PATH)
    for class_name in class_names:
        CLASS_NAMES[class_name] = counter
        counter += 1
    
    return CLASS_NAMES, counter


class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length * self.org_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


""" AdamW Optimizer
Impl copied from PyTorch master
"""

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
    
def plot_fig(test_img, scores, gts, threshold, save_dir, class_name, length):    
    num = length #len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        # cbar_ax = fig_img.add_axes(rect)
        # locator = mticker.MultipleLocator(10)
        # cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        # cb.update_ticks()
        # cb.ax.tick_params(labelsize=8)
        # font = {
        #     'family': 'serif',
        #     'color': 'black',
        #     'weight': 'normal',
        #     'size': 8,
        # }
        # cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()
            
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x    


def plot_anomaly_score_distributions(scores: dict, ground_truths_list, save_folder, class_name):
    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 100000

    for k, v in scores.items():
        
        layer_score = np.stack(v, axis=0)
        normal_score = layer_score[ground_truths == 0]
        abnormal_score = layer_score[ground_truths != 0]

        plt.clf()
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        with plt.style.context(['classic']):
            sns.histplot(np.random.choice(normal_score, N_COUNT), color="green", bins=50, label='${d(p_n)}$',
                         stat='probability', alpha=.75)
            sns.histplot(np.random.choice(abnormal_score, N_COUNT), color="red", bins=50, label='${d(p_a)}$',
                         stat='probability', alpha=.75)

        plt.xlim([0, 5])

        save_path = os.path.join(save_folder, f'distributions_{class_name}_{k}.jpg')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)


valid_feature_visualization_methods = ['TSNE', 'PCA']
def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):
    assert method in valid_feature_visualization_methods
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)

    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')


def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label == l, 0],
                      feat_proj[label == l, 1],
                      feat_proj[label == l, 2], s=5)

    return ax1


def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label == l, 0],
                    feat_proj[label == l, 1], s=5)

    return ax1

def metric_cal(scores, gt_list, gt_mask_list, cal_pro=False):
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list, dtype=int)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    # print('INFO: image ROCAUC: %.3f' % (img_roc_auc))

    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list, dtype=int)
    # print(gt_mask.shape, scores.shape)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    # calculate pro
    if cal_pro:
        pro_auc_score = cal_pro_metric(gt_mask_list, scores, fpr_thresh=0.3)
    else:
        pro_auc_score = 0

    return img_roc_auc, per_pixel_rocauc, pro_auc_score, threshold

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score



def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x:     Samples from the domain of the function to integrate
               Need to be sorted in ascending order. May contain the same value
               multiple times. In that case, the order of the corresponding
               y values will affect the integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
               determined by interpolating between its neighbors. Must not lie
               outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("""WARNING: Not all x and y values passed to trapezoid(...)
                 are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def generate_toy_dataset(num_images, image_width, image_height, gt_size):
    """
    Generate a toy dataset to test the evaluation script.

    Args:
        num_images:   Number of images that the toy dataset contains.
        image_width:  Width of the dataset images in pixels.
        image_height: Height of the dataset images in pixels.
        gt_size:      Size of rectangular ground truth regions that are
                      artificially generated on the dataset images.

    Returns:
        anomaly_maps:     List of numpy arrays that contain random anomaly maps.
        ground_truth_map: Corresponding list of numpy arrays that specify a
                          rectangular ground truth region at a random location.
    """
    # Fix a random seed for reproducibility.
    np.random.seed(1338)

    # Create synthetic evaluation data with random anomaly scores and
    # simple ground truth maps.
    anomaly_maps = []
    ground_truth_maps = []
    for _ in range(num_images):
        # Sample a random anomaly maps.
        anomaly_map = np.random.random((image_height, image_width))

        # Construct a fixed ground truth maps.
        ground_truth_map = np.zeros((image_height, image_width))
        ground_truth_map[0:gt_size, 0:gt_size] = 1

        anomaly_maps.append(anomaly_map)
        ground_truth_maps.append(ground_truth_map)

    return anomaly_maps, ground_truth_maps


class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing
    thresholds.
    """

    def __init__(self, anomaly_scores):
        """
        Initialize the module.

        Args:
            anomaly_scores: List of all anomaly scores within the ground truth
                            component as numpy array.
        """
        # Keep a sorted list of all anomaly scores within the component.
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # Pointer to the anomaly score where the current threshold divides
        # the component into OK / NOK pixels.
        self.index = 0

        # The last evaluated threshold.
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Compute the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.

        Args:
            threshold: Threshold to compute the region overlap.

        Returns:
            Region overlap for the specified threshold.
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        # Increase the index until it points to an anomaly score that is just
        # above the specified threshold.
        while (self.index < len(self.anomaly_scores) and
               self.anomaly_scores[self.index] <= threshold):
            self.index += 1

        # Compute the fraction of component pixels that are correctly segmented
        # as anomalous.
        return 1.0 - self.index / len(self.anomaly_scores)


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extract anomaly scores for each ground truth connected component
    as well as anomaly scores for each potential false positive pixel from
    anomaly maps.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a
                           real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
                           contain binary-valued ground truth labels for each
                           pixel.
                           0 indicates that a pixel is anomaly-free.
                           1 indicates that a pixel contains an anomaly.

    Returns:
        ground_truth_components: A list of all ground truth connected components
                                 that appear in the dataset. For each component,
                                 a sorted list of its anomaly scores is stored.

        anomaly_scores_ok_pixels: A sorted list of anomaly scores of all
                                  anomaly-free pixels of the dataset. This list
                                  can be used to quickly select thresholds that
                                  fix a certain false positive rate.
    """
    # Make sure an anomaly map is present for each ground truth map.
    assert len(anomaly_maps) == len(ground_truth_maps)

    # Initialize ground truth components and scores of potential fp pixels.
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(
        len(ground_truth_maps) * ground_truth_maps[0].size)

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    # Collect anomaly scores within each ground truth region and for all
    # potential fp pixels.
    print("Collect anomaly scores ..")
    ok_index = 0
    for gt_map, prediction in tqdm(zip(ground_truth_maps, anomaly_maps),
                                        total=len(ground_truth_maps)):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)

        # Store all potential fp scores.
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = \
            prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        # Fetch anomaly scores within each GT component.
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(
                GroundTruthComponent(component_scores))

    # Sort all potential false positive scores.
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    print(f"Sort {len(anomaly_scores_ok_pixels)} anomaly scores ..")
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro2(anomaly_maps, ground_truth_maps, num_thresholds):
    """
    Compute the PRO curve at equidistant interpolation points for a set of
    anomaly maps with corresponding ground truth maps. The number of
    interpolation points can be set manually.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a
                           real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
                           contain binary-valued ground truth labels for each
                           pixel.
                           0 indicates that a pixel is anomaly-free.
                           1 indicates that a pixel contains an anomaly.

        num_thresholds:    Number of thresholds to compute the PRO curve.

    Returns:
        fprs: List of false positive rates.
        pros: List of correspoding PRO values.
    """
    # Fetch sorted anomaly scores.
    ground_truth_components, anomaly_scores_ok_pixels = \
        collect_anomaly_scores(anomaly_maps, ground_truth_maps)

    # Select equidistant thresholds.
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1,
                                      num=num_thresholds, dtype=int)

    print("Compute PRO curve..")
    fprs = [1.0]
    pros = [1.0]
    for pos in tqdm.tqdm(threshold_positions):

        threshold = anomaly_scores_ok_pixels[pos]

        # Compute the false positive rate for this threshold.
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # Compute the PRO value for this threshold.
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        print("len(ground_truth_components):" + str(len(ground_truth_components)))
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)

    # Return (FPR/PRO) pairs in increasing FPR order.
    fprs = fprs[::-1]
    pros = pros[::-1]

    return fprs, pros

import pandas as pd
from skimage import measure
def compute_pro(masks, amaps, num_th=200):

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = df["pro"]
    return pro_auc

def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    print("pro auc:{}".format(pro_auc_score))
    return pro_auc_score

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
def MDS_visualization(X, output_path='imgs/MDS.jpg'):
    # multidimension scaling
    
    model2d=MDS(n_components=2, 
          metric=True, 
          n_init=4, 
          max_iter=300, 
          verbose=0, 
          eps=0.001, 
          n_jobs=None, 
          random_state=42, 
          dissimilarity='euclidean')

    batch_size = X.shape[0]
    y = torch.randn(batch_size,)
    
    X_trans = model2d.fit_transform(X)
    # print('The new shape of X: ',X_trans.shape)
    # print('No. of Iterations: ', model2d.n_iter_)
    # print('Stress: ', model2d.stress_)

    # Dissimilarity matrix contains distances between data points in the original high-dimensional space
    # print('Dissimilarity Matrix: ', model2d.dissimilarity_matrix_)
    # Embedding contains coordinates for data points in the new lower-dimensional space
    # print('Embedding: ', model2d.embedding_)

    # Create a scatter plot
    fig = px.scatter(None, x=X_trans[:,0], y=X_trans[:,1], opacity=1, color=y)

    # Change chart background color
    fig.update_layout(dict(plot_bgcolor = 'white'))

    # Update axes lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')

    # Set figure title
    fig.update_layout(title_text="MDS Transformation")

    # Update marker size
    fig.update_traces(marker=dict(size=5,
                                line=dict(color='black', width=0.2)))
    fig.write_image(output_path)
    
def mask_image(image, box):
    x1, y1, x2, y2 = box
    image[:, y1:y2, x1:x2] = 0
    return image
    
def crop_image_CHW(image, coord, K):
    h, w = coord
    if image.shape[1] < 64 or image.shape[2] < 64:
        raise ("image shape not fit: {}|{}".format(image.shape[1], image.shape[2]))
    return image[:, h: h + K, w: w + K]
    
def gen_coord_position(height, width, patch_size=64):
    
    J = patch_size//4
    w1 = np.random.randint(1, width-patch_size-1)
    h1 = np.random.randint(1, height-patch_size-1)    

    pos = np.random.randint(12)
    h_dir, w_dir = pos_to_diff[pos]

    h_del, w_del = np.random.randint(J, size=2)

    K3_4 = 3*patch_size//4
    h_diff = h_dir * (h_del + K3_4)
    w_diff = w_dir * (w_del + K3_4)

    w2 = w1 + int(round(w_diff, 0))
    h2 = h1 + int(round(h_diff, 0))
    
    w2 = np.clip(w2, 0, width-patch_size-1)
    h2 = np.clip(h2, 0, height-patch_size-1)
        
    p1, p2 = (w1,h1), (w2,h2)
    
    return p1, p2, pos    
    
def select_another_image_from_same_class(img1_path):
    root_dir = os.path.dirname(img1_path)

    image_path_list = glob(os.path.join(root_dir, "*"))    

    new_idx = random.randint(1,len(image_path_list)-1)
    while image_path_list[new_idx] == img1_path:
        new_idx = random.randint(1,len(image_path_list)-1)

    img2_path = image_path_list[new_idx]
    return img1_path, img2_path
    
def select_image_from_other_class(img1_path):
    # classes = 
    path1 = os.path.dirname(img1_path)
    root_dir = os.path.dirname(os.path.dirname(path1))
    classes_list = glob(os.path.join(root_dir, "*"))
    class_name = os.path.dirname(img1_path).split("/")[-2]
    
    new_idx = random.randint(1,len(classes_list)-1)
    while classes_list[new_idx] == class_name:
        new_idx = random.randint(1,len(classes_list)-1)
        
    image_list = glob(os.path.join(classes_list[new_idx], 'normal', '*.*'))
    if len(image_list) < 5:
        print(classes_list[new_idx])
    image_idx = random.randint(1, len(image_list)-1)
     
    img3_path = image_list[image_idx]
    return img1_path, img3_path

def adjust_learning_rate(optimizers, init_lrs, epoch, args):
    """Decay the learning rate based on schedule"""
    for i in range(3):
        cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = cur_lr
    
def save_output_image(image, image_name):
    image = torch.sigmoid(image)
    save_image(image, image_name)
    
def save_grid_image(images, image_name):
    img = make_grid(images, normalize=True)
    save_image(img, image_name)
    
def load_hdf5(infile, keys):
    with h5py.File(infile, 'r') as f:
        return {key: f[key][:] for key in keys}
    
def write_hdf5(outfile, arr_dict):
    with h5py.File(outfile, 'w') as f:
        for key in arr_dict.keys():
            f.create_dataset(key, data=arr_dict[key])


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

def print_model_layer_content(model):
    content = model.state_dict()
    for _ , layer_name in enumerate(content):
        print(layer_name)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        
def training_data_len(PATH):
    training_data_count = 0
    class_names = os.listdir(PATH)
    for class_name in class_names:
        training_data_count += len(os.listdir(os.path.join(PATH, class_name)))
    return training_data_count

def make_permutation(N):
    return [i//2 if (i % 2 == 0) else N+i//2 for i in range(2*N)]

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def get_center(model, device, train_loader):
    train_feature_space = []
    print("Start calculate center...")
    with torch.no_grad():
        for imgs in tqdm(train_loader):
            imgs = torch.cat(imgs, dim=0)
            imgs = imgs.to(device)
            model = model.to(device)
            features = model(imgs)
            train_feature_space.append(features)

        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    return train_feature_space



class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
    
    
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label, colors_per_class):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)
    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, colors_per_class, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
        zip(images, labels, tx, ty),
        desc='Building the T-SNE plot',
        total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label, colors_per_class)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()
    plt.savefig('visualize_tsne_image.png')


def visualize_tsne_points(tx, ty, labels):
    # print('Plotting TSNE image')
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors_per_class = {
        0 : [254, 202, 87],
        1 : [255, 107, 107],
        2 : [10, 189, 227],
        3 : [255, 159, 243],
        4 : [16, 172, 132],
        5 : [128, 80, 128]
    }
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        
        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
        
        # add a scatter plot with the correponding color and label

        ax.scatter(current_tx, current_ty, c=color, label=label)

        # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()
    plt.savefig('visualize_tsne_points.png')


def visualize_tsne2(tsne, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    #visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)

def visualize_tsne(tsne, labels, class_name, verbose=False):
    # tsne = tsne.reshape(labels.shape[0], -1)
    # print(tsne.shape)
    # print(labels.shape)

    cmap = plt.cm.get_cmap("spring")
    colors = ['green', 'red']
    legends = ["normal", "anomaly"]
    markers = ["*", "x"]
    (_, ax) = plt.subplots(1)

    plt.title(label=class_name)
    labels = torch.tensor(labels)
    for label in torch.unique(labels):
        res = tsne[torch.where(labels==label)]
        ax.plot(*res.T, marker=markers[label], linestyle="", ms=5, label=legends[label], color=colors[label])
        ax.legend(loc="best")
    plt.xticks([])
    plt.yticks([])

    plt.savefig('./imgs/'+class_name+'_'+'tsne_output.png')
    plt.close()
    
def plot_tsne(labels, embeds, defect_name = None, save_path = None, **kwargs: Dict[str, Any]):
        """t-SNE visualize

        Args:
            labels (Tensor): labels of test and train
            embeds (Tensor): embeds of test and train
            defect_name ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
            save_path ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
            kwargs (Dict[str, Any]): hyper parameters of t-SNE which will change final result
                n_iter (int): > 250, default = 1000
                learning_rate (float): (10-1000), default = 100
                perplexity (float): (5-50), default = 28
                early_exaggeration (float): change it when not converging, default = 12
                angle (float): (0.2-0.8), default = 0.3
                init (str): "random" or "pca", default = "pca"
        """        
        tsne = TSNE(
            n_components=2, 
            verbose=1, 
            n_iter=kwargs.get("n_iter", 1000),
            learning_rate=kwargs.get("learning_rate", 100),
            perplexity=kwargs.get("perplexity", 28), 
            early_exaggeration=kwargs.get("early_exaggeration", 12),
            angle=kwargs.get("angle", 0.3),
            init=kwargs.get("init", "pca"),
        )
        embeds, labels = shuffle(embeds, labels)
        tsne_results = tsne.fit_transform(embeds)

        cmap = plt.cm.get_cmap("spring")
        colors = np.vstack((np.array([[0, 1. ,0, 1.]]), cmap([0, 256//3, (2*256)//3])))
        legends = ["good", "anomaly"]
        (_, ax) = plt.subplots(1)
        plt.title(f't-SNE: {defect_name}')
        for label in torch.unique(labels):
            res = tsne_results[torch.where(labels==label)]
            ax.plot(*res.T, marker="*", linestyle="", ms=5, label=legends[label], color=colors[label])
            ax.legend(loc="best")
        plt.xticks([])
        plt.yticks([])

        save_images = save_path if save_path else './tnse_results'
        os.makedirs(save_images, exist_ok=True)
        image_path = os.path.join(save_images, defect_name+'_tsne.jpg') if defect_name else os.path.join(save_images, 'tsne.jpg')
        plt.savefig(image_path)
        plt.close()
        return
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']    


def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
