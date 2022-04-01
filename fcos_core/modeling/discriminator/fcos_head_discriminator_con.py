import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal

class FCOSDiscriminator_con(nn.Module):
    def __init__(self, with_GA=False, fusion_cfg ='concat', num_convs=2, in_channels=256, num_classes=2, grad_reverse_lambda=-1.0, grl_applied_domain='both',patch_stride=None):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_con, self).__init__()
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

        # self.num_classes = num_classes - 1
        self.num_classes = num_classes
        self.with_GA = with_GA
        self.fusion_cfg = fusion_cfg

        self.class_cond_map = []
        for i in range(self.num_classes):
            self.class_cond_map.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels+1 if self.fusion_cfg == 'concat' else in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels, 1, kernel_size=3, stride=1,
                        padding=1)
                )
            )

            for i, block in enumerate(self.class_cond_map):
                self.add_module('classifier_cls_{}'.format(i),block)

        if self.with_GA:
            self.cls_logits = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1
            )
        self.patch_stride = patch_stride
        assert patch_stride==None or type(patch_stride)==int, 'wrong format of patch stride'
        for modules in [self.dis_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        for modules in self.class_cond_map:
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


    def forward(self, feature, act_maps):


        # if self.grl_applied_domain == 'both':
        #     feature = self.grad_reverse(feature)
        #     act_maps = self.grad_reverse(act_maps)
        # elif self.grl_applied_domain == 'target':
        #     if domain == 'target':
        #         feature = self.grad_reverse(feature)
        # if self.patch_stride:
        #     feature = self.pool(feature)


        features_s, features_t = feature
        features_s = self.grad_reverse(features_s)
        features_t = self.grad_reverse(features_t)

        act_maps_s, act_maps_t = act_maps
        act_maps_s = self.grad_reverse(act_maps_s)
        act_maps_t = self.grad_reverse(act_maps_t)

        features_s = self.dis_tower(features_s)
        features_t = self.dis_tower(features_t)

        loss = features_t.new_zeros([1])

        for c in range(self.num_classes):
            # x = torch.mul(feature,act_maps[:,c+1,:,:].unsqueeze(1))

            if self.fusion_cfg == 'concat':
                x_cls_s = torch.cat((features_s, act_maps_s[:,c,:,:].unsqueeze(1)),dim=1)
                x_cls_t = torch.cat((features_t, act_maps_t[:,c,:,:].unsqueeze(1)),dim=1)
            elif self.fusion_cfg == 'mul':
                x_cls_s = torch.mul(features_s, act_maps_s[:, c, :, :].unsqueeze(1)).contiguous()
                x_cls_t = torch.mul(features_t, act_maps_t[:, c, :, :].unsqueeze(1)).contiguous()
            elif self.fusion_cfg == 'mul_detached':
                x_cls_s = torch.mul(features_s, act_maps_s[:, c, :, :].unsqueeze(1).detach())
                x_cls_t = torch.mul(features_t, act_maps_t[:, c, :, :].unsqueeze(1).detach())
            else:
                raise KeyError("Unknown fusion config!")

            logits_s = self.class_cond_map[c](x_cls_s)
            logits_t = self.class_cond_map[c](x_cls_t)


            loss_s = self.loss_fn(logits_s,  torch.full(logits_s.shape, self.source_label, dtype=torch.float, device=logits_s.device))
            loss_s += loss_s / self.num_classes


            loss_t = self.loss_fn(logits_t,  torch.full(logits_t.shape, self.target_label, dtype=torch.float, device=logits_t.device))
            loss_t += loss_t / self.num_classes
            loss += (loss_t +loss_s)

        #
        # if self.with_GA:
        #     logits_GA = self.cls_logits(x)
        #     targets = torch.full(logits_GA.shape, target, dtype=torch.float, device=x.device)
        #     loss = 0.5* loss +  self.loss_fn(logits_GA, targets)


        return loss
