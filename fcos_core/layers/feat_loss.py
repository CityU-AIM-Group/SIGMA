import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FeatLoss(nn.Module):
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
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        super(FeatLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)


    def forward(self, features_t):

        loss = 0
        for i, feat in enumerate(features_t):
            B, C, H, W = feat.size()
            target = (feat.abs()==0).float()
            loss += self.loss(feat, target)

        return  loss/len(features_t)