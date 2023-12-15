import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from utils import make_permutation

def deviation_loss(a, b, y_true):
    # print(y_true)
    confidence_margin = 5.
    ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
    
    cos_loss = torch.nn.CosineSimilarity()
    
    loss  = 0
    for item in range(len(a)):
        ## yMAX(0, ref - cos_sim) + (1-y)|cos_sim|
        cos_similarity = torch.mean(1 - cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))

        inlier_loss = torch.abs(cos_similarity)
        dev = (cos_similarity - torch.mean(ref)) / torch.std(ref)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))

        loss += torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

    return loss

class Howeng_Loss(nn.Module):
    def __init__(self, batch_size, views, temperature=0.5):
        super(Howeng_Loss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size 
        self.views = views
        self.N = self.views * self.batch_size
        self.mask = self.mask_correlated_samples()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_matrix = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self):   
        # select positive and set them to 0(FALSE)
        mask = torch.ones((self.N, self.N), dtype=bool)
        
        for i in range(self.N):
            if i+self.batch_size < self.N:
                mask[i, i+self.batch_size] = 0
                mask[i+self.batch_size, i] = 0
                
            if i+2*self.batch_size < self.N:
                mask[i, i+2*self.batch_size] = 0
                mask[i+2*self.batch_size, i] = 0
                
            if i-self.batch_size > self.N:
                mask[i, i-self.batch_size] = 0
                mask[i-self.batch_size, i] = 0

            if i-2*self.batch_size > self.N:
                mask[i, i-2*self.batch_size] = 0
                mask[i-2*self.batch_size, i] = 0

        mask[:, -self.batch_size:] = 1
        mask[-self.batch_size:, :] = 1
        mask = mask.fill_diagonal_(0)
        return mask
        
    def forward(self, z_i, z_j, z_k):
        
        z1 = F.normalize(z_i)
        z2 = F.normalize(z_j)
        z3 = F.normalize(z_k)
        z = torch.cat((z1, z2, z3), dim=0)
        
        sim = self.similarity_matrix(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # positives
        self.mask = self.mask.fill_diagonal_(1)
        # print(self.mask.shape, sim.shape)
        positives = sim[~self.mask.bool()].to('cuda')
        
        positives = positives.view(self.batch_size*(self.views-1), 2)
        # print(positives.shape)

        # negatives
        self.mask = self.mask.fill_diagonal_(0)
        negatives = sim[self.mask.bool()].to('cuda')
        # print(negatives.shape)
        # negatives_front = negatives[:(self.N-1-self.batch_size)*(self.batch_size*(self.views-1))]
        negatives_front  = negatives[:self.batch_size*(self.views-1) * (self.N-1-2)]
        # print(negatives_front.shape)
        negatives_front = negatives_front.view(self.batch_size*(self.views-1), (self.N-1-2))
        # print(negatives_front.shape)
        
        # negatives_back  = negatives[(self.N-1-self.batch_size)*(self.batch_size*(self.views-1)):]
        negatives_back  = negatives[-self.batch_size*(self.N-1):]        
        # print(negatives_back.shape)
        negatives_back  = negatives_back.view(self.N-(self.batch_size*(self.views-1)), self.N-1)        
        # print(negatives_back.shape)
        # print(negatives_front.shape, negatives_back.shape)

        labels = torch.zeros(self.N, dtype=torch.long).to('cuda')

        # print(positives.shape, negatives.shape)
        logits = torch.cat([positives, negatives_front], dim=1)
        logits = torch.cat([logits, negatives_back], dim=0)
        
        loss = self.criterion(logits, labels)        
        loss /= self.N
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class Contrastive_Loss(nn.Module):
    def __init__(self, temperature=0.4):
        super(Contrastive_Loss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        n = z_i.shape[0]
        z1 = F.normalize(z_i)
        z2 = F.normalize(z_j)
        z = torch.cat([z1, z2])
        logits = torch.mm(z, z.T).div(self.temperature)
        logits.fill_diagonal_(float('-inf'))
        labels = torch.tensor(list(range(n, 2*n)) + list(range(n)), device=logits.device)
        # print(logits.shape, labels.shape)
        loss = F.cross_entropy(logits, labels)
        
        return loss

class NT_XentLoss(nn.Module):
    def __init__(self, T):
        super(NT_XentLoss, self).__init__()
        self.temperature = T

    def forward(self, i, j, s):
        # s : 2N x 2N tensor, cosine similarity matrix
        s = torch.exp(s/self.temperature)
        numerator = s[i, j]
        denominator = torch.sum(s[i, :], dim=-1) - s[i, i]

        return -torch.log(numerator/denominator)

    def info_nce_loss(z_i, z_j, criterion):
        N = z_i.size(0)  # batch size
        
        #z_i, z_j =  [N, 512], [N,512]
        zT = torch.cat([z_i, z_j], dim=0)  # [2N, 512]

        perm = make_permutation(N)  # [0, N, 1, N+1, ..., N-1, 2N-1]
        zT = zT[perm, :]
        z_norm = torch.norm(zT, dim=-1).unsqueeze(-1)  # [2N, 1]

        z = zT.transpose(0, 1)  # [512, 2N]
        z_normT = z_norm.transpose(0, 1)  # [1, 2N]

        zTz = torch.mm(zT, z)  # [2N, 2N]
        zzT_norm = torch.mm(z_norm, z_normT)  # [2N, 2N]

        s = zTz / zzT_norm  # [2N, 2N]

        loss = 0.
        for k in range(N):
            loss += criterion(2 * k, 2 * k + 1, s) + criterion(2 * k + 1, 2 * k, s)

        return loss/(2 * N)

#https://github.com/sthalles/SimCLR/blob/master/simclr.py
class Info_NCE_loss(nn.Module):
    def __init__(self, args, temperature):
        super(Info_NCE_loss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.branch = 2
        
    def forward(self, z_i, z_j):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.branch)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        norm_z_i = F.normalize(z_i, dim=-1)
        norm_z_j = F.normalize(z_j, dim=-1)
        # features = F.normalize(features, dim=1)
        features = torch.cat([norm_z_i, norm_z_j], dim=0)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            self.branch * self.args.batch_size, self.branch * self.args.batch_size)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.temperature
        return logits, labels

# ----------REFERENCES------------- #
# https://github.com/talreiss/Mean-Shifted-Anomaly-Detection/blob/f976ddf043edf8e0e4265f700bc5abaea1dfe4ea/main.py#L10s
# https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss
    
def CosLoss(data1, data2, Mean=True):
    data2 = data2.detach()
    cos = nn.CosineSimilarity(dim=1)
    if Mean:
        return -cos(data1, data2).mean()
    else:
        return -cos(data1, data2)
    
    
class SPD_Loss(nn.Module):
    def __init__(self, temperature=0.2, epsilon=0.1):
        super(SPD_Loss, self).__init__()
        self.epsilon = epsilon
        self.temperature = temperature
        
    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def forward(self, anchor, positive, negative):
        
        query, positive, negative = self.normalize(anchor, positive, negative)
        batch_size = anchor.size(0)
        
        SPD_loss = (F.cosine_similarity(x1=anchor, x2=negative, dim=1, eps=1e-8) - F.cosine_similarity(x1=anchor, x2=positive, dim=1, eps=1e-8)).mean()

        positive_logit = torch.sum(query * positive, dim=1, keepdim=True)
        negative_logits = query @ negative.transpose(-2, -1)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        NCE_loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        total_loss = NCE_loss + self.epsilon * SPD_loss
        return total_loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
