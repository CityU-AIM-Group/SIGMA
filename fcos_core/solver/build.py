# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, name):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if name == 'backbone':
            lr = cfg.SOLVER.BACKBONE.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BACKBONE.BASE_LR * cfg.SOLVER.BACKBONE.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'fcos':
            lr = cfg.SOLVER.FCOS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.FCOS.BASE_LR * cfg.SOLVER.FCOS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'roi_head':
            lr = cfg.SOLVER.FCOS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.FCOS.BASE_LR * cfg.SOLVER.FCOS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        elif name == 'middle_head':
            lr = cfg.SOLVER.MIDDLE_HEAD.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.MIDDLE_HEAD.BASE_LR * cfg.SOLVER.MIDDLE_HEAD.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'discriminator':
            lr = cfg.SOLVER.DIS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.DIS.BASE_LR * cfg.SOLVER.DIS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        else:
            raise AssertionError('here')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # if cfg.SOLVER.OPTIMIZER == 'SGD':
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    # elif cfg.SOLVER.OPTIMIZER == 'Adam':
    #     optimizer = torch.optim.Adam(params, lr)



    return optimizer


def make_lr_scheduler(cfg, optimizer, name):
    if name == 'backbone':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.BACKBONE.STEPS,
            cfg.SOLVER.BACKBONE.GAMMA,
            warmup_factor=cfg.SOLVER.BACKBONE.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.BACKBONE.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.BACKBONE.WARMUP_METHOD,
        )
    elif name == 'fcos':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.FCOS.STEPS,
            cfg.SOLVER.FCOS.GAMMA,
            warmup_factor=cfg.SOLVER.FCOS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.FCOS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.FCOS.WARMUP_METHOD,
        )
    elif name == 'middle_head':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.MIDDLE_HEAD.STEPS,
            cfg.SOLVER.MIDDLE_HEAD.GAMMA,
            warmup_factor=cfg.SOLVER.MIDDLE_HEAD.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.MIDDLE_HEAD.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.MIDDLE_HEAD.WARMUP_METHOD,
        )
    elif name == 'discriminator':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.DIS.STEPS,
            cfg.SOLVER.DIS.GAMMA,
            warmup_factor=cfg.SOLVER.DIS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.DIS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.DIS.WARMUP_METHOD,
        )
    elif name == 'roi_head':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.FCOS.STEPS,
            cfg.SOLVER.FCOS.GAMMA,
            warmup_factor=cfg.SOLVER.FCOS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.FCOS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.FCOS.WARMUP_METHOD,
        )
    else:
        raise AssertionError('here')