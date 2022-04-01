import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal, FocalLoss

class FCOSDiscriminator_CA(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, center_aware_weight=0.0, center_aware_type='ca_loss', grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_CA, self).__init__()

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

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_no_reduce = nn.BCEWithLogitsLoss(reduction='none')

        # hyperparameters
        # assert center_aware_type == 'ca_loss' or center_aware_type == 'ca_feature'
        self.center_aware_weight = center_aware_weight
        self.center_aware_type = center_aware_type

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain


    def forward(self, feature, target, score_map=None, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        # Generate cneter-aware map
        box_cls_map = score_map["box_cls"].clone().sigmoid()
        centerness_map = score_map["centerness"].clone().sigmoid()

        n, c, h, w = box_cls_map.shape
        maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        box_cls_map = maxpooling(box_cls_map)

        # Normalize the center-aware map
        atten_map = (self.center_aware_weight * box_cls_map * centerness_map).sigmoid()


        # Compute loss
        # Center-aware loss (w/ GRL)
        if self.center_aware_type == 'ca_loss':
            if self.grl_applied_domain == 'both':
                feature = self.grad_reverse(feature)
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature = self.grad_reverse(feature)

            # Forward
            x = self.dis_tower(feature)
            x = self.cls_logits(x)

            # Computer loss
            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn_no_reduce(x, target)
            loss = torch.mean(atten_map * loss)

        # Center-aware feature (w/ GRL)
        elif self.center_aware_type == 'ca_feature':
            if self.grl_applied_domain == 'both':
                feature = self.grad_reverse(atten_map * feature)
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature = self.grad_reverse(atten_map * feature)

            # Forward
            x = self.dis_tower(feature)
            x = self.cls_logits(x)

            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            loss = self.loss_fn(x, target)

        elif self.center_aware_type == 'focal':
            if self.grl_applied_domain == 'both':
                feature = self.grad_reverse(atten_map * feature)
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature = self.grad_reverse(atten_map * feature)

            # Forward
            x = self.dis_tower(feature)
            x = self.cls_logits(x)

            target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            # print(x, target)
            # input()
            loss = FocalLoss(x, target)
            # loss = self.focal_loss(x, target)
        # print('center')
        # print(x, target)

        return loss
