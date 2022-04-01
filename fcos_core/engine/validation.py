# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from fcos_core.data.datasets.evaluation import validate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
# from fcos_core.engine.trainer import foward_detector
from fcos_core.structures.image_list import to_image_list
def _foward_detector(cfg, model, images, targets=None, return_maps=False):

    # For faster rcnn
    with_rcnn = not cfg.MODEL.RPN_ONLY

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]
    images = to_image_list(images)
    features = model_backbone(images.tensors)

    if cfg.MODEL.MIDDLE_HEAD.CONDGRAPH_ON:
        middle_head = model["middle_head"]
        features, return_act_maps = middle_head(images, features)

    if with_rcnn:
        model_roi_head = model["roi_head"]
        proposals, proposal_losses = model_fcos(
            images, features, targets=targets)
        feats, proposals, roi_head_loss = model_roi_head(features, proposals, targets)

    else:
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets, return_maps=return_maps)
        # inference
    return proposals

def compute_on_dataset(cfg, model, data_loader, device, timer=None):
    # model.eval
    for k in model:
        model[k].eval()

    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output = _foward_detector(cfg, model, images, targets=None)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def _inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")

    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(cfg, model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return validate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
