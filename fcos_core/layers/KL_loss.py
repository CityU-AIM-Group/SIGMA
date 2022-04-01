import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class KLLoss(nn.Module):
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
        super(KLLoss, self).__init__()


        # self.KLLOSS = nn.KLDivLoss()
        # self.KLLOSS = F.kl_div()


    # def forward(self, batch, global_prototype, label_indx, mode= 'prototype'):
    #     if mode == 'prototype':
    #         label_indx = label_indx.bool()
    #         batch = batch[label_indx]
    #         target = global_prototype[label_indx]
    #         KL_loss=  F.kl_div(batch.softmax(-1).log().detach(), target.softmax(-1))
    #     elif mode == 'nodes':
    #         n, c = batch.size()
    #         label_indx = label_indx.long()
    #         target = global_prototype[label_indx]
    #         # KL_loss = self.KLLOSS(target.softmax(-1).log().detach(), batch.softmax(-1))
    #         # KL_loss = F.kl_div( batch.log_softmax(-1), target.softmax(-1).detach())
    #         KL_loss = F.kl_div( batch.softmax(-1).log(), target.softmax(-1).detach())
    #
    #
    #     else: raise KeyError('unknown KL loss mode')
    #     return  KL_loss
    def forward(self, batch_exist, global_exist):


        # indx = batch_prototype.sum(dim=-1) != 0
        # import ipdb
        # ipdb.set_trace()
        # Remove unexisted pairs
        # batch_exist = batch_prototype[indx]
        # global_exist = global_prototype[indx]

        KL_loss = F.kl_div(batch_exist.softmax(-1).log(), global_exist.softmax(-1).detach())

        # if mode == 'prototype':
        #     label_indx = label_indx.bool()
        #     batch = batch[label_indx]
        #     target = global_prototype[label_indx]
        #     KL_loss=  F.kl_div(batch.softmax(-1).log().detach(), target.softmax(-1))
        # elif mode == 'nodes':
        #     n, c = batch.size()
        #     label_indx = label_indx.long()
        #     target = global_prototype[label_indx]
        #     # KL_loss = self.KLLOSS(target.softmax(-1).log().detach(), batch.softmax(-1))
        #     # KL_loss = F.kl_div( batch.log_softmax(-1), target.softmax(-1).detach())
        #     KL_loss = F.kl_div( batch.softmax(-1).log(), target.softmax(-1).detach())
        #
        #
        # else: raise KeyError('unknown KL loss mode')
        return  KL_loss