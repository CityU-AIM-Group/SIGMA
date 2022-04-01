# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip
import random
import argparse
import os
import numpy as np
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn, build_middle_head
from fcos_core.modeling.discriminator import FCOSDiscriminator, FCOSDiscriminator_CA, FCOSDiscriminator_out, FCOSDiscriminator_con
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

from fcos_core.utils.metric_logger import (
    MetricLogger, TensorboardLogger)

def train(cfg, local_rank, distributed, test_only=False, use_tensorboard=False):


    with_DA = cfg.MODEL.DA_ON

    ##########################################################################
    ############################# Initial MODEL ##############################
    ##########################################################################
    MODEL = {}
    device = torch.device(cfg.MODEL.DEVICE)
    backbone = build_backbone(cfg).to(device)
    # print(backbone)
    # if cfg.MODEL.MIDDLE_HEAD.CONDGRAPH_ON:
    if cfg.MODEL.MIDDLE_HEAD_CFG == 'GM_HEAD':
        middle_head = build_middle_head(cfg, backbone.out_channels).to(device)
    fcos = build_rpn(cfg, backbone.out_channels).to(device)


    ##########################################################################
    #################### Initial Optimizer and Scheduler #####################
    ##########################################################################
    optimizer = {}
    scheduler = {}


    optimizer["backbone"] = make_optimizer(cfg, backbone, name='backbone')
    optimizer["fcos"] = make_optimizer(cfg, fcos, name='fcos')
    if cfg.MODEL.MIDDLE_HEAD_CFG == 'GM_HEAD':
        optimizer["middle_head"] = make_optimizer(cfg, middle_head, name='middle_head')

    scheduler["backbone"] = make_lr_scheduler(cfg, optimizer["backbone"], name='backbone')
    scheduler["fcos"] = make_lr_scheduler(cfg, optimizer["fcos"], name='fcos')
    if cfg.MODEL.MIDDLE_HEAD_CFG == 'GM_HEAD':
        scheduler["middle_head"] = make_lr_scheduler(cfg, optimizer["middle_head"], name='middle_head')

    if with_DA:
        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P7_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P7,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P6_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P6,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P5_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P5,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P4_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P4,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P3_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P3,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P7_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P7,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P6_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P6,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P5_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P5,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P4_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P4,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P3_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P3,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_OUT:
            out_da_channels = 0
            if cfg.MODEL.ADV.OUTPUT_REG_DA:
                out_da_channels += 4
            if cfg.MODEL.ADV.OUTPUT_CLS_DA:
                if cfg.MODEL.ADV.OUTMAP_OP == 'attreg':
                    out_da_channels += 0
                elif cfg.MODEL.ADV.OUTMAP_OP == 'maxpool':
                    out_da_channels += 1
                else:
                    if cfg.MODEL.ATSS_ON:
                        out_da_channels += cfg.MODEL.ATSS.NUM_CLASSES
                        out_da_channels -= 1
                    elif cfg.MODEL.FCOS_ON:
                        out_da_channels += cfg.MODEL.FCOS.NUM_CLASSES
                        out_da_channels -= 1
                    else:
                        raise NotImplementedError
            if cfg.MODEL.ADV.OUTPUT_CENTERNESS_DA:
                out_da_channels += 1
            assert out_da_channels != 0, "Output alignment should have at least 1 channels !"

            dis_P7_OUT = FCOSDiscriminator_out(
                num_convs=cfg.MODEL.ADV.CA_DIS_P7_NUM_CONVS,
                in_channels=out_da_channels,  # TODO: Should change this as a new cfg ?
                cls_map_pre=cfg.CLS_MAP_PRE,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P7,
                out_weight=cfg.MODEL.ADV.OUT_WEIGHT,
                out_loss=cfg.MODEL.ADV.OUT_LOSS,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                outmap_op=cfg.MODEL.ADV.OUTMAP_OP,
                output_reg_da=cfg.MODEL.ADV.OUTPUT_REG_DA,
                output_cls_da=cfg.MODEL.ADV.OUTPUT_CLS_DA,
                output_centerness_da=cfg.MODEL.ADV.OUTPUT_CENTERNESS_DA,
                base_dis_tower=cfg.MODEL.ADV.BASE_DIS_TOWER).to(device)
            dis_P6_OUT = FCOSDiscriminator_out(
                num_convs=cfg.MODEL.ADV.CA_DIS_P6_NUM_CONVS,
                in_channels=out_da_channels,
                cls_map_pre=cfg.CLS_MAP_PRE,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P6,
                out_weight=cfg.MODEL.ADV.OUT_WEIGHT,
                out_loss=cfg.MODEL.ADV.OUT_LOSS,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                outmap_op=cfg.MODEL.ADV.OUTMAP_OP,
                output_reg_da=cfg.MODEL.ADV.OUTPUT_REG_DA,
                output_cls_da=cfg.MODEL.ADV.OUTPUT_CLS_DA,
                output_centerness_da=cfg.MODEL.ADV.OUTPUT_CENTERNESS_DA,
                base_dis_tower=cfg.MODEL.ADV.BASE_DIS_TOWER).to(device)
            dis_P5_OUT = FCOSDiscriminator_out(
                num_convs=cfg.MODEL.ADV.CA_DIS_P5_NUM_CONVS,
                in_channels=out_da_channels,
                cls_map_pre=cfg.CLS_MAP_PRE,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P5,
                out_weight=cfg.MODEL.ADV.OUT_WEIGHT,
                out_loss=cfg.MODEL.ADV.OUT_LOSS,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                outmap_op=cfg.MODEL.ADV.OUTMAP_OP,
                output_reg_da=cfg.MODEL.ADV.OUTPUT_REG_DA,
                output_cls_da=cfg.MODEL.ADV.OUTPUT_CLS_DA,
                output_centerness_da=cfg.MODEL.ADV.OUTPUT_CENTERNESS_DA,
                base_dis_tower=cfg.MODEL.ADV.BASE_DIS_TOWER).to(device)
            dis_P4_OUT = FCOSDiscriminator_out(
                num_convs=cfg.MODEL.ADV.CA_DIS_P4_NUM_CONVS,
                in_channels=out_da_channels,
                cls_map_pre=cfg.CLS_MAP_PRE,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P4,
                out_weight=cfg.MODEL.ADV.OUT_WEIGHT,
                out_loss=cfg.MODEL.ADV.OUT_LOSS,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                outmap_op=cfg.MODEL.ADV.OUTMAP_OP,
                output_reg_da=cfg.MODEL.ADV.OUTPUT_REG_DA,
                output_cls_da=cfg.MODEL.ADV.OUTPUT_CLS_DA,
                output_centerness_da=cfg.MODEL.ADV.OUTPUT_CENTERNESS_DA,
                base_dis_tower=cfg.MODEL.ADV.BASE_DIS_TOWER).to(device)
            dis_P3_OUT = FCOSDiscriminator_out(
                num_convs=cfg.MODEL.ADV.CA_DIS_P3_NUM_CONVS,
                in_channels=out_da_channels,
                cls_map_pre=cfg.CLS_MAP_PRE,
                grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P3,
                out_weight=cfg.MODEL.ADV.OUT_WEIGHT,
                out_loss=cfg.MODEL.ADV.OUT_LOSS,
                grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                outmap_op=cfg.MODEL.ADV.OUTMAP_OP,
                output_reg_da=cfg.MODEL.ADV.OUTPUT_REG_DA,
                output_cls_da=cfg.MODEL.ADV.OUTPUT_CLS_DA,
                output_centerness_da=cfg.MODEL.ADV.OUTPUT_CENTERNESS_DA,
                base_dis_tower=cfg.MODEL.ADV.BASE_DIS_TOWER).to(device)
        if cfg.MODEL.ADV.USE_DIS_CON:
            if cfg.MODEL.ADV.USE_DIS_P7_CON:
                dis_P7_CON = FCOSDiscriminator_con(
                    with_GA = cfg.MODEL.ADV.CON_WITH_GA,
                    fusion_cfg = cfg.MODEL.ADV.CON_FUSUIN_CFG,
                    num_convs=cfg.MODEL.ADV.CON_NUM_SHARED_CONV_P7,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P7,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                    num_classes = cfg.MODEL.FCOS.NUM_CLASSES,
                ).to(device)

            if cfg.MODEL.ADV.USE_DIS_P6_CON:
                dis_P6_CON = FCOSDiscriminator_con(
                    with_GA=cfg.MODEL.ADV.CON_WITH_GA,
                    fusion_cfg=cfg.MODEL.ADV.CON_FUSUIN_CFG,
                    num_convs=cfg.MODEL.ADV.CON_NUM_SHARED_CONV_P6,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P6,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                    num_classes=cfg.MODEL.FCOS.NUM_CLASSES,
                ).to(device)
            if cfg.MODEL.ADV.USE_DIS_P5_CON:
                dis_P5_CON = FCOSDiscriminator_con(
                    with_GA=cfg.MODEL.ADV.CON_WITH_GA,
                    fusion_cfg=cfg.MODEL.ADV.CON_FUSUIN_CFG,
                    num_convs=cfg.MODEL.ADV.CON_NUM_SHARED_CONV_P5,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P5,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                    num_classes=cfg.MODEL.FCOS.NUM_CLASSES,
                ).to(device)
            if cfg.MODEL.ADV.USE_DIS_P4_CON:
                dis_P4_CON = FCOSDiscriminator_con(
                    with_GA=cfg.MODEL.ADV.CON_WITH_GA,
                    fusion_cfg=cfg.MODEL.ADV.CON_FUSUIN_CFG,
                    num_convs=cfg.MODEL.ADV.CON_NUM_SHARED_CONV_P4,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P4,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                    num_classes=cfg.MODEL.FCOS.NUM_CLASSES,
                ).to(device)
            if cfg.MODEL.ADV.USE_DIS_P3_CON:
                dis_P3_CON = FCOSDiscriminator_con(
                    with_GA=cfg.MODEL.ADV.CON_WITH_GA,
                    fusion_cfg=cfg.MODEL.ADV.CON_FUSUIN_CFG,
                    num_convs=cfg.MODEL.ADV.CON_NUM_SHARED_CONV_P3,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P3,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN,
                    num_classes=cfg.MODEL.FCOS.NUM_CLASSES,
                ).to(device)

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                optimizer["dis_P7"] = make_optimizer(cfg, dis_P7, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                optimizer["dis_P6"] = make_optimizer(cfg, dis_P6, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                optimizer["dis_P5"] = make_optimizer(cfg, dis_P5, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                optimizer["dis_P4"] = make_optimizer(cfg, dis_P4, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                optimizer["dis_P3"] = make_optimizer(cfg, dis_P3, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                optimizer["dis_P7_CA"] = make_optimizer(cfg, dis_P7_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                optimizer["dis_P6_CA"] = make_optimizer(cfg, dis_P6_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                optimizer["dis_P5_CA"] = make_optimizer(cfg, dis_P5_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                optimizer["dis_P4_CA"] = make_optimizer(cfg, dis_P4_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                optimizer["dis_P3_CA"] = make_optimizer(cfg, dis_P3_CA, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_OUT:
            optimizer["dis_P7_OUT"] = make_optimizer(cfg, dis_P7_OUT, name='discriminator')
            optimizer["dis_P6_OUT"] = make_optimizer(cfg, dis_P6_OUT, name='discriminator')
            optimizer["dis_P5_OUT"] = make_optimizer(cfg, dis_P5_OUT, name='discriminator')
            optimizer["dis_P4_OUT"] = make_optimizer(cfg, dis_P4_OUT, name='discriminator')
            optimizer["dis_P3_OUT"] = make_optimizer(cfg, dis_P3_OUT, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_CON:
            optimizer["dis_P7_CON"] = make_optimizer(cfg, dis_P7_CON, name='discriminator')
            optimizer["dis_P6_CON"] = make_optimizer(cfg, dis_P6_CON, name='discriminator')
            optimizer["dis_P5_CON"] = make_optimizer(cfg, dis_P5_CON, name='discriminator')
            optimizer["dis_P4_CON"] = make_optimizer(cfg, dis_P4_CON, name='discriminator')
            optimizer["dis_P3_CON"] = make_optimizer(cfg, dis_P3_CON, name='discriminator')

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                scheduler["dis_P7"] = make_lr_scheduler(cfg, optimizer["dis_P7"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                scheduler["dis_P6"] = make_lr_scheduler(cfg, optimizer["dis_P6"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                scheduler["dis_P5"] = make_lr_scheduler(cfg, optimizer["dis_P5"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                scheduler["dis_P4"] = make_lr_scheduler(cfg, optimizer["dis_P4"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                scheduler["dis_P3"] = make_lr_scheduler(cfg, optimizer["dis_P3"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                scheduler["dis_P7_CA"] = make_lr_scheduler(cfg, optimizer["dis_P7_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                scheduler["dis_P6_CA"] = make_lr_scheduler(cfg, optimizer["dis_P6_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                scheduler["dis_P5_CA"] = make_lr_scheduler(cfg, optimizer["dis_P5_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                scheduler["dis_P4_CA"] = make_lr_scheduler(cfg, optimizer["dis_P4_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                scheduler["dis_P3_CA"] = make_lr_scheduler(cfg, optimizer["dis_P3_CA"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_OUT:
            scheduler["dis_P7_OUT"] = make_lr_scheduler(cfg, optimizer["dis_P7_OUT"], name='discriminator')
            scheduler["dis_P6_OUT"] = make_lr_scheduler(cfg, optimizer["dis_P6_OUT"], name='discriminator')
            scheduler["dis_P5_OUT"] = make_lr_scheduler(cfg, optimizer["dis_P5_OUT"], name='discriminator')
            scheduler["dis_P4_OUT"] = make_lr_scheduler(cfg, optimizer["dis_P4_OUT"], name='discriminator')
            scheduler["dis_P3_OUT"] = make_lr_scheduler(cfg, optimizer["dis_P3_OUT"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_CON:
            scheduler["dis_P7_CON"] = make_lr_scheduler(cfg, optimizer["dis_P7_CON"], name='discriminator')
            scheduler["dis_P6_CON"] = make_lr_scheduler(cfg, optimizer["dis_P6_CON"], name='discriminator')
            scheduler["dis_P5_CON"] = make_lr_scheduler(cfg, optimizer["dis_P5_CON"], name='discriminator')
            scheduler["dis_P4_CON"] = make_lr_scheduler(cfg, optimizer["dis_P4_CON"], name='discriminator')
            scheduler["dis_P3_CON"] = make_lr_scheduler(cfg, optimizer["dis_P3_CON"], name='discriminator')

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                MODEL["dis_P7"] = dis_P7
            if cfg.MODEL.ADV.USE_DIS_P6:
                MODEL["dis_P6"] = dis_P6
            if cfg.MODEL.ADV.USE_DIS_P5:
                MODEL["dis_P5"] = dis_P5
            if cfg.MODEL.ADV.USE_DIS_P4:
                MODEL["dis_P4"] = dis_P4
            if cfg.MODEL.ADV.USE_DIS_P3:
                MODEL["dis_P3"] = dis_P3
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                MODEL["dis_P7_CA"] = dis_P7_CA
            if cfg.MODEL.ADV.USE_DIS_P6:
                MODEL["dis_P6_CA"] = dis_P6_CA
            if cfg.MODEL.ADV.USE_DIS_P5:
                MODEL["dis_P5_CA"] = dis_P5_CA
            if cfg.MODEL.ADV.USE_DIS_P4:
                MODEL["dis_P4_CA"] = dis_P4_CA
            if cfg.MODEL.ADV.USE_DIS_P3:
                MODEL["dis_P3_CA"] = dis_P3_CA
        if cfg.MODEL.ADV.USE_DIS_OUT:
            MODEL["dis_P7_OUT"] = dis_P7_OUT
            MODEL["dis_P6_OUT"] = dis_P6_OUT
            MODEL["dis_P5_OUT"] = dis_P5_OUT
            MODEL["dis_P4_OUT"] = dis_P4_OUT
            MODEL["dis_P3_OUT"] = dis_P3_OUT
        if cfg.MODEL.ADV.USE_DIS_CON:
            MODEL["dis_P7_CON"] = dis_P7_CON
            MODEL["dis_P6_CON"] = dis_P6_CON
            MODEL["dis_P5_CON"] = dis_P5_CON
            MODEL["dis_P4_CON"] = dis_P4_CON
            MODEL["dis_P3_CON"] = dis_P3_CON

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
        fcos = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fcos)

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P7)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P6)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P5)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P4)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P3)

        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P7_CA)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P6_CA)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P5_CA)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P4_CA)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P3_CA)

        if cfg.MODEL.ADV.USE_DIS_OUT:
            dis_P7_OUT = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P7_OUT)
            dis_P6_OUT = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P6_OUT)
            dis_P5_OUT = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P5_OUT)
            dis_P4_OUT = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P4_OUT)
            dis_P3_OUT = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P3_OUT)
    ##########################################################################
    ######################## DistributedDataParallel #########################
    ##########################################################################

    if distributed:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )
        if cfg.MODEL.MIDDLE_HEAD_CFG == 'GM_HEAD':
            middle_head = torch.nn.parallel.DistributedDataParallel(
                middle_head, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False
            )
        fcos = torch.nn.parallel.DistributedDataParallel(
            fcos, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = torch.nn.parallel.DistributedDataParallel(
                    dis_P7, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = torch.nn.parallel.DistributedDataParallel(
                    dis_P6, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = torch.nn.parallel.DistributedDataParallel(
                    dis_P5, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = torch.nn.parallel.DistributedDataParallel(
                    dis_P4, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = torch.nn.parallel.DistributedDataParallel(
                    dis_P3, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )

        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P7_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P6_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P5_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P4_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P3_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )

        if cfg.MODEL.ADV.USE_DIS_OUT:
            dis_P7_OUT = torch.nn.parallel.DistributedDataParallel(dis_P7_OUT, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P6_OUT = torch.nn.parallel.DistributedDataParallel(dis_P6_OUT, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P5_OUT = torch.nn.parallel.DistributedDataParallel(dis_P5_OUT, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P4_OUT = torch.nn.parallel.DistributedDataParallel(dis_P4_OUT, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P3_OUT = torch.nn.parallel.DistributedDataParallel(dis_P3_OUT, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        if cfg.MODEL.ADV.USE_DIS_CON:
            dis_P7_CON = torch.nn.parallel.DistributedDataParallel(dis_P7_CON, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P6_CON = torch.nn.parallel.DistributedDataParallel(dis_P6_CON, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P5_CON = torch.nn.parallel.DistributedDataParallel(dis_P5_CON, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P4_CON = torch.nn.parallel.DistributedDataParallel(dis_P4_CON, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            dis_P3_CON = torch.nn.parallel.DistributedDataParallel(dis_P3_CON, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    ##########################################################################
    ########################### Save MODEL to Dict ###########################
    ##########################################################################
    MODEL["backbone"] = backbone
    MODEL["fcos"] = fcos
    if cfg.MODEL.MIDDLE_HEAD_CFG == 'GM_HEAD':
        MODEL["middle_head"] = middle_head

    ##########################################################################
    ################################ Training ################################
    ##########################################################################
    arguments = {}
    arguments["iteration"] = 0
    if with_DA:
        arguments["use_dis_global"] = cfg.MODEL.ADV.USE_DIS_GLOBAL
        arguments["use_dis_ca"] = cfg.MODEL.ADV.USE_DIS_CENTER_AWARE
        arguments["use_dis_out"] = cfg.MODEL.ADV.USE_DIS_OUT
        arguments["use_dis_con"] = cfg.MODEL.ADV.USE_DIS_CON
        arguments["ga_dis_lambda"] = cfg.MODEL.ADV.GA_DIS_LAMBDA
        arguments["ca_dis_lambda"] = cfg.MODEL.ADV.CA_DIS_LAMBDA
        arguments["out_dis_lambda"] = cfg.MODEL.ADV.OUT_DIS_LAMBDA
        arguments["con_dis_lambda"] = cfg.MODEL.ADV.CON_DIS_LAMBDA

        arguments["use_feature_layers"] = []
        arguments["use_feature_layers"].append("P7") if cfg.MODEL.ADV.USE_DIS_P7 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P6") if cfg.MODEL.ADV.USE_DIS_P6 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P5") if cfg.MODEL.ADV.USE_DIS_P5 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P4") if cfg.MODEL.ADV.USE_DIS_P4 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P3") if cfg.MODEL.ADV.USE_DIS_P3 else arguments["use_feature_layers"]


    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, MODEL, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, load_dis=True, load_opt_sch=False)
    # arguments.update(extra_checkpoint_data)

    # Initial dataloader (both target and source domain)
    data_loader = {}
    data_loader["source"] = make_data_loader_source(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    if cfg.SOLVER.ADAPT_VAL_ON:
        data_loader["val"] = make_data_loader(
            cfg,
            is_train=False,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

    if with_DA:
        data_loader["target"] = make_data_loader_target(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        dir = os.path.join(cfg.OUTPUT_DIR,'tensorboard_logs/')

        if not os.path.exists(dir):
            mkdir(dir)
        meters = TensorboardLogger(
            log_dir=dir,
            start_iter=arguments['iteration'],
            delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")
    print(MODEL)

    if not test_only:
        do_train(
            MODEL,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            cfg,
            distributed,
            meters,
        )
    return MODEL


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def run_test(cfg, MODEL, distributed):
    if distributed:
        MODEL["backbone"] = MODEL["backbone"].module
        MODEL["fcos"] = MODEL["fcos"].module

    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    # print(MODEL)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            MODEL,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="./configs/SIGMA/sigma_vgg16_cityscapace_to_foggy.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Test the input MODEL directly, without training",
        action="store_true",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final MODEL",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use_tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX installed)",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # Check if domain adaption
    # assert cfg.MODEL.DA_ON, "Domain Adaption"

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    setup_seed(1234)
    MODEL = train(cfg, args.local_rank, args.distributed, args.test_only,args.use_tensorboard)

    if not args.skip_test:
        run_test(cfg, MODEL, args.distributed)


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    main()
