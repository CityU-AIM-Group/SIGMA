import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CosineLoss(nn.Module):
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
        super(CosineLoss, self).__init__()

        self.CosineLoss = nn.CosineEmbeddingLoss(margin)
        # self.KLLOSS = nn.KLDivLoss(size_average=True, reduce=True)


    def forward(self, batch, glb, label_indx):
        label_indx = label_indx.bool()
        batch = batch[label_indx]
        glb = glb[label_indx]
        sim_loss = self.CosineLoss(batch, glb.detach(), torch.full((batch.size(0),1), 1, dtype=torch.float, device=batch.device).squeeze())
        KL_loss= self.KLLOSS(glb.softmax(-1).log().detach(), batch.softmax(-1))
        # print(KL_loss)

        return sim_loss, KL_loss