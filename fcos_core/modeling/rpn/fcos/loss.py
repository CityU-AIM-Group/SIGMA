# --------------------------------------------------------
# SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection (CVPR22-ORAL)
# Modified by Wuyang Li
# This file contains the original fcos loss computation and the node sampling of SIGMA
# --------------------------------------------------------
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss, SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
import os
from sklearn import *
import time
INF = 100000000

class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """
    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        locations = locations.cuda()

        object_sizes_of_interest= object_sizes_of_interest.cuda()
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):

            targets_per_im = targets[im_i]

            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox.cuda()
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def replace_targets(self,locations, box_cls, box_regression, centerness, targets):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        reg_targets_flatten = []
        labels, reg_targets = self.prepare_targets(locations, targets)
        tmp = []

        for l in range(len(labels)):
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            tmp.append(reg_targets[l].size(0))
        reg_targets_flatten = torch.cat(reg_targets_flatten,dim=0)
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
        centerness_targets_list = []
        k = 0
        for i in tmp:
            centerness_targets_list.append(centerness_targets[k:k+i])
            k += i

        box_cls_gt = []
        box_reg_gt = []
        box_ctr_gt = []

        for l in range(len(labels)):
            n, c, h, w = box_cls[l].size()
            if c >len(labels):
                c=c-1
            lb = F.one_hot(labels[l].reshape(-1), 9)[:,1:].float()
            box_cls_gt.append(lb.reshape(n,h,w,c).permute(0,3,1,2).cuda())
            box_reg_gt.append(reg_targets[l].reshape(-1).reshape(n,h,w,4).permute(0,3,1,2).cuda())
            box_ctr_gt.append(centerness_targets_list[l].reshape(-1).reshape(n,h,w,1).permute(0,3,1,2).float().cuda())
        return box_cls_gt, box_reg_gt, box_ctr_gt

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss

def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator

class PrototypeComputation(object):
    """
    This class computes the FCOS losses.
    """
    def __init__(self, cfg):
        self.num_class = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.num_class_fgbg = cfg.MODEL.FCOS.NUM_CLASSES
        self.cfg =cfg.clone()
        self.class_threshold = cfg.SOLVER.MIDDLE_HEAD.PLABEL_TH
        self.num_nodes_per_class = cfg.MODEL.MIDDLE_HEAD.GM.NUM_NODES_PER_LVL_SR
        self.num_nodes_per_lvl = cfg.MODEL.MIDDLE_HEAD.GM.NUM_NODES_PER_LVL_TG
        self.bg_ratio = cfg.MODEL.MIDDLE_HEAD.GM.BG_RATIO
    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            # reg_targets_level_first.append(
            #     torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            # )
        return labels_level_first
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, features, targets):

        if locations: # Sampling in the source domain
            N, C, _, _ = features[0].size()
            labels = self.prepare_targets(locations, targets)

            pos_points = []
            pos_labels = []
            neg_points = []
            for l in range(len(labels)):
                pos_indx =  labels[l].reshape(-1) > 0
                neg_indx =  labels[l].reshape(-1) == 0

                # Sparse sampling to save GPU memory
                pos_nodes_all = features[l].permute(0, 2, 3, 1).reshape(-1, C)[pos_indx]
                pos_labels_all = labels[l][pos_indx]
                step = len(pos_labels_all) //self.num_nodes_per_class
                if step>1:
                    pos_points.append(pos_nodes_all[::step])
                    pos_labels.append(pos_labels_all[::step])
                else:
                    pos_points.append(pos_nodes_all)
                    pos_labels.append(pos_labels_all)
                num_pos = len(pos_points[-1])

                # Sampling Background Nodes
                if self.cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG:
                    neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx]
                    if len(labels[l][pos_indx]) > len(labels[l][neg_indx]):
                        neg_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx])
                    else:
                        # neg_indx = list(np.floor(np.linspace(0,len(labels[l][neg_indx])-2, (len(labels[l][pos_indx])))/8).astype(int))
                        neg_indx = list(np.floor(np.linspace(0,len(labels[l][neg_indx])-2, num_pos//self.bg_ratio)))
                        neg_points.append(neg_points_temp[neg_indx])

            pos_points = torch.cat(pos_points,dim=0)
            pos_labels = torch.cat(pos_labels,dim=0)

            if self.cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG:
                neg_points = torch.cat(neg_points, dim=0)
                neg_labels = pos_labels.new_zeros((neg_points.size(0)))
                pos_points = torch.cat([neg_points, pos_points] ,dim=0)
                pos_labels = torch.cat([neg_labels, pos_labels] )

            return pos_points, pos_labels, pos_labels.new_ones(pos_labels.shape).long()

        else: # Sampling in the target domain
            act_maps_lvl_first = targets
            N, C, _, _ = features[0].size()
            N, Cls, _, _ = targets[0].size()
            neg_points =[]
            pos_plabels = []
            pos_points = []
            pos_weight = []
            for l, feature in enumerate(features):
                act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(-1, self.num_class)
                conf_pos_indx = (act_maps > self.class_threshold[0]).sum(dim=-1).bool()
                neg_indx = (act_maps < 0.05).sum(dim=-1).bool()
                # Balanced sampling BG pixels
                if conf_pos_indx.any():
                    act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(-1, self.num_class)
                    if conf_pos_indx.sum() > self.num_nodes_per_lvl :
                        raw_features = features[l].permute(0, 2, 3, 1).reshape(-1, C)[conf_pos_indx]
                        twice_indx = torch.randperm(raw_features.size(0))[:100]
                        pos_points.append(raw_features[twice_indx])
                        scores, indx = act_maps[conf_pos_indx, :].max(-1)
                        scores = scores[twice_indx]
                        indx = indx[twice_indx]
                    else:
                        pos_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[conf_pos_indx])
                        scores, indx = act_maps[conf_pos_indx,:].max(-1)

                    # pos_plabels.append(act_maps[conf_pos_indx,:].argmax(dim=-1) + 1)
                    pos_plabels.append(indx + 1)
                    pos_weight.append(scores.detach())
                    # neg_indx = ~conf_pos_indx
                    neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx]
                    num_pos = len(scores)
                    neg_indx_new = list(np.floor(np.linspace(0, (neg_indx.sum()- 2).item(), (num_pos//self.bg_ratio))).astype(int))
                    neg_points.append(neg_points_temp[neg_indx_new])
            # print(end-start)
            if len(pos_points)>0:
                pos_points = torch.cat(pos_points,dim=0)
                pos_plabels = torch.cat(pos_plabels,dim=0)
                neg_points = torch.cat(neg_points, dim=0)
                neg_plabels = pos_plabels.new_zeros((neg_points.size(0)))
                pos_weight = torch.cat(pos_weight, dim=0)
                neg_weight = pos_weight.new_ones(neg_points.size(0)) * 0.5
                points = torch.cat([neg_points, pos_points], dim=0)
                plabels = torch.cat([neg_plabels, pos_plabels])

                loss_weight = torch.cat([neg_weight, pos_weight])
                return points, plabels, loss_weight.long()
            else:
                return None, None, None

def make_prototype_evaluator(cfg):
    prototype_evaluator = PrototypeComputation(cfg)
    return prototype_evaluator
