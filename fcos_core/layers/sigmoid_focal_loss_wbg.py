import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
def see(data, name='default'):
    print('#################################',name,'#################################')
    print('max: ', torch.max(data))
    print('mean: ', torch.mean(data))
    print('min: ', torch.min(data))
    print('##########################################################################')


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
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs.softmax(dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs =(P*class_mask).sum(1).view(-1,1)

        
        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        # print(batch_loss)
        if self.size_average:
            loss =batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# 支持多分类和二分类
# class FocalLoss(nn.Module):
#     """
#     This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
#     'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
#       Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
#     :param num_class:
#     :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
#     :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
#             focus on hard misclassified example
#     :param smooth: (float,double) smooth value when cross entropy
#     :param balance_index: (int) balance class index, should be specific when alpha is float
#     :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
#     """
#
#     def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.num_class = num_class
#         self.alpha = alpha
#         self.gamma = gamma
#         self.smooth = smooth
#         self.size_average = size_average
#
#         if self.alpha is None:
#             self.alpha = torch.ones(self.num_class, 1)
#         elif isinstance(self.alpha, (list, np.ndarray)):
#             assert len(self.alpha) == self.num_class
#             self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
#             self.alpha = self.alpha / self.alpha.sum()
#         elif isinstance(self.alpha, float):
#             alpha = torch.ones(self.num_class, 1)
#             alpha = alpha * (1 - self.alpha)
#             alpha[balance_index] = self.alpha
#             self.alpha = alpha
#         else:
#             raise TypeError('Not support alpha type')
#
#         if self.smooth is not None:
#             if self.smooth < 0 or self.smooth > 1.0:
#                 raise ValueError('smooth value should be in [0,1]')
#
#     def forward(self, input, target):
#         logit = F.softmax(input, dim=1)
#         if logit.dim() > 2:
#             # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#             logit = logit.view(logit.size(0), logit.size(1), -1)
#             logit = logit.permute(0, 2, 1).contiguous()
#             logit = logit.view(-1, logit.size(-1))
#         target = target.view(-1, 1)
#
#         # N = input.size(0)
#         # alpha = torch.ones(N, self.num_class)
#         # alpha = alpha * (1 - self.alpha)
#         # alpha = alpha.scatter_(1, target.long(), self.alpha)
#         epsilon = 1e-10
#         alpha = self.alpha
#         if alpha.device != input.device:
#             alpha = alpha.to(input.device)
#
#         idx = target.cpu().long()
#         one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         if one_hot_key.device != logit.device:
#             one_hot_key = one_hot_key.to(logit.device)
#
#         if self.smooth:
#             one_hot_key = torch.clamp(
#                 one_hot_key, self.smooth, 1.0 - self.smooth)
#         pt = (one_hot_key * logit).sum(1) + epsilon
#         logpt = pt.log()
#
#         gamma = self.gamma
#
#         alpha = alpha[idx]
#         loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
#
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.clamp(torch.sigmoid(_input), min=0.00001)
        # pt = torch.clamp(pt, max=0.99999)
        pt = _input
        alpha = self.alpha

        # pos = torch.nonzero(target[:,1] > 0).squeeze(1).numel()
        # print(pos)

        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction =='pos':
            loss = torch.sum(loss) / (2*pos)


        return loss