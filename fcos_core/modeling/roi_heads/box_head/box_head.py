# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .graph_matching_head import build_graph_matching_head
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg,in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg,in_channels)
        self.predictor = make_roi_box_predictor(cfg,in_channels *2)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.with_middle_head = cfg.MODEL.MIDDLE_HEAD.CONDGRAPH_ON
        if self.with_middle_head:
            self.middle_head = build_graph_matching_head(cfg, 2048)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        #[2, 1024, 40, 50]
        # numerical value: [0 to 56.63] mean: 0.241

        det_loss = {}
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}, x, None


        if self.with_middle_head:
            loss_classifier, loss_box_reg, domain_labels, class_gt = self.loss_evaluator(
                [class_logits], [box_regression], return_labels=True
            )
            loss_middle_head = self.middle_head(features, x, class_gt, class_logits, domain_labels)
            det_loss.update(loss_middle_head)
        else:
            loss_classifier, loss_box_reg, _ = self.loss_evaluator(
                [class_logits], [box_regression]
            )

        # calculate loss

        det_loss.update(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

        if self.training:
            with torch.no_grad():
                da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)

        da_ins_feas = self.feature_extractor(features, da_proposals)
        class_logits, box_regression = self.predictor(da_ins_feas)

        _, _, da_ins_labels = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        return (
            x,
            proposals,
            det_loss,
            da_ins_feas,
            da_ins_labels
        )

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    # return ROIBoxHead(cfg, in_channels)
    return ROIBoxHead(cfg, in_channels)
