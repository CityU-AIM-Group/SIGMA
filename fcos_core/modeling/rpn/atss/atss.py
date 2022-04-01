import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_atss_postprocessor
from .loss import make_atss_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d
from ..anchor_generator import make_anchor_generator_atss


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS.ANCHOR_STRIDES[0]
            l = w * (anchors_cx - gt_boxes[:, 0]) / anchors_w
            t = w * (anchors_cy - gt_boxes[:, 1]) / anchors_h
            r = w * (gt_boxes[:, 2] - anchors_cx) / anchors_w
            b = w * (gt_boxes[:, 3] - anchors_cy) / anchors_h
            targets = torch.stack([l, t, r, b], dim=1)
        elif self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            TO_REMOVE = 1  # TODO remove
            ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
            gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)
            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS.ANCHOR_STRIDES[0]
            x1 = anchors_cx - preds[:, 0] / w * anchors_w
            y1 = anchors_cy - preds[:, 1] / w * anchors_h
            x2 = anchors_cx + preds[:, 2] / w * anchors_w
            y2 = anchors_cy + preds[:, 3] / w * anchors_h
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            anchors = anchors.to(preds.dtype)

            TO_REMOVE = 1  # TODO remove
            widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            dx = preds[:, 0::4] / wx
            dy = preds[:, 1::4] / wy
            dw = preds[:, 2::4] / ww
            dh = preds[:, 3::4] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=math.log(1000. / 16))
            dh = torch.clamp(dh, max=math.log(1000. / 16))

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(preds)
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes

def save_feat(feat, name='./loco_sim10k/source.pt'):
    print(feat[0].size())
    feat = torch.cat(feat,dim=0).cpu()
    print(feat.size())
    torch.save(feat, name)

class ATSSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ATSSHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ATSS.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.ATSS.ASPECT_RATIOS) * cfg.MODEL.ATSS.SCALES_PER_OCTAVE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.ATSS.NUM_CONVS):
            if self.cfg.MODEL.ATSS.USE_DCN_IN_TOWER and \
                    i == cfg.MODEL.ATSS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, num_anchors * 1, kernel_size=3, stride=1,
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
        prior_prob = cfg.MODEL.ATSS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            assert num_anchors == 1, "regressing from a point only support num_anchors == 1"
            torch.nn.init.constant_(self.bbox_pred.bias, 4)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        self.buffer_fpn = []
        self.buffer_out = []

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        output_towers = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)

            box_tower = self.bbox_tower(feature)

            # output_towers.append(torch.nn.functional.adaptive_avg_pool2d(torch.cat((cls_tower,box_tower),dim=1),(1,1)).squeeze())

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
                bbox_pred = F.relu(bbox_pred)
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(box_tower))


        # print(output_towers[-1].size())
        # output = torch.cat()

        # self.buffer_fpn.append(torch.nn.functional.adaptive_avg_pool2d(x[-1], (1, 1)).squeeze())
        #
        # self.buffer_out.append(output_towers[-1])


        # if len(self.buffer) == 200:
        # if len(self.buffer_fpn) == 125:
        #     save_feat(self.buffer_fpn, './loco_cs/baseline_target_fpn.pt')
        #     save_feat(self.buffer_out, './loco_cs/baseline_target_out.pt')
            # import os
            # os._exit(0)

        # if len(self.buffer) == 500:
        #     save_feat(self.buffer, './loco_sim10k/baseline_source_fpn_4k.pt')
        #     import os
        #     os._exit(0)
        return logits, bbox_reg, centerness


class ATSSModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ATSSModule, self).__init__()
        self.cfg = cfg
        self.head = ATSSHead(cfg, in_channels)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder)
        self.anchor_generator = make_anchor_generator_atss(cfg)

    def forward(self, images, features, targets=None, return_maps=False,act_maps=None):
        box_cls, box_regression, centerness = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors, return_maps)
        else:
            return self._forward_test(box_cls, box_regression, centerness, anchors)

    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors, return_maps=False):
        # print('anchors on the smallest level: ', anchors[0][4].bbox)
        # input()
        score_maps = {
            "box_cls": box_cls,
            "box_regression": box_regression,
            "centerness": centerness
            # "anchors": anchors
        }
        # for name in score_maps:
        #     print(name)
        #     print(len(score_maps[name]))
        #     for tensor in score_maps[name]:
        #         print(tensor.size())
        # input()
        # box_cls
        # torch.Size([1, 1, 92, 168])
        # torch.Size([1, 1, 46, 84])
        # torch.Size([1, 1, 23, 42])
        # torch.Size([1, 1, 12, 21])
        # torch.Size([1, 1, 6, 11])

        # box_regression
        # torch.Size([1, 4, 92, 168])
        # torch.Size([1, 4, 46, 84])
        # torch.Size([1, 4, 23, 42])
        # torch.Size([1, 4, 12, 21])
        # torch.Size([1, 4, 6, 11])

        # centerness
        # torch.Size([1, 1, 92, 168])
        # torch.Size([1, 1, 46, 84])
        # torch.Size([1, 1, 23, 42])
        # torch.Size([1, 1, 12, 21])
        # torch.Size([1, 1, 6, 11])
        if targets is not None:
            loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
                box_cls, box_regression, centerness, targets, anchors
            )
            losses = {
                "loss_cls": loss_box_cls,
                "loss_reg": loss_box_reg,
                "loss_centerness": loss_centerness
            }
        else:
            losses = {
                "zero": 0.0 * sum(0.0 * torch.sum(x) for x in box_cls)\
                        +0.0 * sum(0.0 * torch.sum(x) for x in box_regression)\
                        +0.0 * sum(0.0 * torch.sum(x) for x in centerness)\
            }
        if return_maps:
            return None, losses, score_maps
        return None, losses, None

    def _forward_test(self, box_cls, box_regression, centerness, anchors):
        boxes = self.box_selector_test(box_cls, box_regression, centerness, anchors)
        return boxes, {}, None


def build_atss(cfg, in_channels):
    return ATSSModule(cfg, in_channels)