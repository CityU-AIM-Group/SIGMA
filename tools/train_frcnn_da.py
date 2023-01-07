# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.solver import make_lr_scheduler, make_lr_scheduler_frcnn
from fcos_core.solver import make_optimizer_frcnn
from fcos_core.engine.inference_frcnn import inference
from fcos_core.engine.trainer_frcnn import do_train, do_da_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
import numpy as np
import random
def train(cfg, local_rank, distributed):

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer_frcnn(cfg, model)
    scheduler = make_lr_scheduler_frcnn(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD



    if cfg.MODEL.DA_ON:
        source_data_loader = make_data_loader_source(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        val_data_loader = make_data_loader(
            cfg,
            is_train=False,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        target_data_loader = make_data_loader_target(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

        setup_seed(1234)
        do_da_train(
            model,
            source_data_loader,
            target_data_loader,
            val_data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            cfg,
        )
    else:
        data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )


        val_data_loader = make_data_loader(
            cfg,
            is_train=False,
            is_distributed=distributed)

        do_train(
            model,
            data_loader,
            val_data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            cfg,
        )

    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test(cfg, model, distributed):
    if distributed:
        model = model.module
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
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
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
        # default="configs/osda_faster_rcnn/osda_faster_rcnn_R_50_C4_10_10_oracle.yaml",
        # default="configs/osda_faster_rcnn/osda_faster_rcnn_R_50_C4_15_5.yaml",
        default="configs/SIGMA_frcnn/base.yaml",
        # default="configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_voc_to_clpart.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
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

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
