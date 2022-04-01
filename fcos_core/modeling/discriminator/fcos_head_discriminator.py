import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal





class FCOSDiscriminator(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, grl_applied_domain='both',patch_stride=None):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator, self).__init__()
        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        self.patch_stride = patch_stride
        assert patch_stride==None or type(patch_stride)==int, 'wrong format of patch stride'
        if self.patch_stride:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=patch_stride, padding=1)
            # patch = nn.AdaptiveAvgPool2d()
        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain

        self.source_label = 1.0
        self.target_label = 0.0


    def forward(self, feature, domain='source'):


        # if self.grl_applied_domain == 'both':
        #     feature = self.grad_reverse(feature)
        # elif self.grl_applied_domain == 'target':
        #     if domain == 'target':
        #         feature = self.grad_reverse(feature)
        features_s, features_t = feature

        features_s = self.grad_reverse(features_s)
        features_t = self.grad_reverse(features_t)

        x_s = self.cls_logits(self.dis_tower(features_s))
        x_t = self.cls_logits(self.dis_tower(features_t))

        target_source = torch.full(x_s.shape, self.source_label, dtype=torch.float, device=x_s.device)
        target_target = torch.full(x_t.shape, self.target_label, dtype=torch.float, device=x_t.device)
        # target = torch.cat([target_source, target_target], dim=0)
        loss_s = self.loss_fn(x_s, target_source)
        loss_t = self.loss_fn(x_t, target_target)

        return loss_s + loss_t
