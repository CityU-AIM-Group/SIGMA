import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d=256):
        super(Affinity, self).__init__()
        self.d = d

        self.fc_M = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )

        # self.project_sr = nn.Linear(256, 256,bias=False)
        # self.project_tg = nn.Linear(256, 256,bias=False)
        self.project_sr = nn.Linear(256, 256,bias=False)
        self.project_tg = nn.Linear(256, 256,bias=False)
        self.reset_parameters()


    def reset_parameters(self):

        for i in self.fc_M:
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, std=0.01)
                nn.init.constant_(i.bias, 0)


        nn.init.normal_(self.project_sr.weight, std=0.01)
        nn.init.normal_(self.project_tg.weight, std=0.01)

        # The common GM design doesn;t work!!
        # stdv = 1. / math.sqrt(self.d)
        # self.A.data.uniform_(-stdv, stdv)
        # self.A.data += torch.eye(self.d).cuda()
        # nn.init.normal_(self.project_2.weight, std=0.01)
        # nn.init.normal_(self.project2.weight, std=0.01)
        # nn.init.constant_(i.bias, 0)
    def forward(self, X, Y):

        X = self.project_sr(X)
        Y = self.project_tg(Y)

        N1, C = X.size()
        N2, C = Y.size()

        X_k = X.unsqueeze(1).expand(N1, N2, C)
        Y_k = Y.unsqueeze(0).expand(N1, N2, C)
        M = torch.cat([X_k, Y_k], dim=-1)
        M = self.fc_M(M).squeeze()

        # The common GM design doesn;t work!!

        # M = self.affinity_pred(M[None,]).squeeze()
        # M_r = self.fc_M(M_r).squeeze()
        # M = torch.matmul(X, (self.A + self.A.transpose(0, 1).contiguous()) / 2)
        # M = torch.matmul(M, Y.transpose(0, 1).contiguous())


        return M
