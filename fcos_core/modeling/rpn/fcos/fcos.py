import math
import torch
import torch.nn.functional as F
from torch import nn
from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator
from fcos_core.layers import Scale
import os
import matplotlib.pyplot as plt
import ipdb


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS_CLS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

        for i in range(cfg.MODEL.FCOS.NUM_CONVS_REG):
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        self.with_reg_ctr = cfg.MODEL.FCOS.REG_CTR_ON
        self.mode = cfg.TEST.MODE

        self.debug_cfg = cfg.MODEL.DEBUG_CFG
        if self.debug_cfg:
            from fcos_core.vis_tools import VIS_TOOLS
            self.debuger = VIS_TOOLS()

    def forward(self, x, act_maps=False):
        logits = []
        bbox_reg = []
        centerness = []

        # output_towers =[]
        for l, feature in enumerate(x):

            if self.mode !='light' or self.training:
                cls_tower = self.cls_tower(feature)
                logits.append(self.cls_logits(cls_tower))

            if self.with_reg_ctr:
                reg_tower = self.bbox_tower(feature)
                centerness.append(self.centerness(reg_tower))
                bbox_reg.append(torch.exp(self.scales[l](
                    self.bbox_pred(reg_tower)
                )))
            else:
                centerness.append(self.centerness(cls_tower))
                bbox_reg.append(torch.exp(self.scales[l](
                    self.bbox_pred(self.bbox_tower(feature))
                )))


        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)
        box_selector_test = make_fcos_postprocessor(cfg)
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.mode = cfg.TEST.MODE
        self.with_ctr = cfg.MODEL.MIDDLE_HEAD.GM.WITH_CTR


        # DEBUG SETTINGS

        # self.cnt = 0 # FOR SAVING FEATURES
        self.debug_cfg = cfg.MODEL.DEBUG_CFG
        if self.debug_cfg:
            from fcos_core.vis_tools import VIS_TOOLS
            self.debugger = VIS_TOOLS()


    def forward(self, images, features, targets=None, return_maps=False, act_maps= None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        box_cls, box_regression, centerness = self.head(features, act_maps)

        locations = self.compute_locations(features)
        if not self.training:
            if self.mode == 'light':
                box_cls = []
                for i in range(len(centerness)):
                    box_cls.append(act_maps[i][:, 1:, :, :])
            elif  self.mode == 'precision':
                for i in range(len(centerness)):
                    box_cls[i] = (0.5*box_cls[i].sigmoid() + 0.5 * act_maps[i][:, 1:, :, :])

        # if self.debug_cfg:
        #     for i in range(len(centerness)):
        #         if  'CLS_MAP' in self.debug_cfg:
        #             self.debugger.debug_draw_maps(box_cls[i].sigmoid(), i, name='classification', exit=False)
        #         if 'CNT_MAP' in self.debug_cfg:
        #             self.debugger.debug_draw_maps(centerness[i].sigmoid(), i, name='centerness', exit=False)
        #     # ONLY DEBUG ONE BATCH
        #     os._exit(0)



        if self.training and targets:
            return self._forward_train_source(
                locations, box_cls,
                box_regression,
                centerness, targets, return_maps
            )
        elif self.training :
            return self._forward_target(
                box_cls, box_regression,
                centerness
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )




    def _forward_train_source(self, locations, box_cls, box_regression, centerness, targets, return_maps=False):

        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }


        return None, losses, None

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):

        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}, None

    def _forward_target(self, box_cls, box_regression, centerness):
        scores = []
        for i, cls_logits in enumerate(box_cls):
            if self.with_ctr:

                mask = (centerness[i].sigmoid()>0.5).float()
                scores.append((cls_logits.sigmoid() * mask).detach())
            else:
                scores.append(cls_logits.sigmoid().detach())


        losses = {
            "zero": 0.0 * sum(0.0 * torch.sum(x) for x in box_cls)
                    +0.0 * sum(0.0 * torch.sum(x) for x in box_regression)
                    +0.0 * sum(0.0 * torch.sum(x) for x in centerness)}

        return scores, losses, None

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)

