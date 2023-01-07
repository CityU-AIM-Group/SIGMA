# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size
from fcos_core.utils.metric_logger import MetricLogger
import warnings
from fcos_core.engine.inference_frcnn import inference


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
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
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0 and iteration != 0:
            if val_data_loader is not None:
                model.eval()
                val_results = validataion(cfg, model, val_data_loader[0], distributed=False)

                meter_AP50 = val_results['map']
                # logger.info('[validation mAP] AP: {}, AP50: {}'.format(meter_AP,meter_AP50))
                logger.info('[validation mAP] AP: {}, AP50: {}'.format(meter_AP50, meter_AP50))

                if meter_AP50 > max_AP50_bank:
                    max_AP50_bank = meter_AP50
                    checkpointer.save("best_model_updated_{}_{:07d}".format(meter_AP50, iteration), **arguments)
                model.train()

            else:

                checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_da_train(
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
        cfg
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    print(model)
    start_training_time = time.time()
    end = time.time()
    max_AP50_bank = cfg.SOLVER.INITIAL_AP50
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(
            zip(source_data_loader, target_data_loader), start_iter):

        data_time = time.time() - end
        arguments["iteration"] = iteration
        images = (source_images + target_images).to(device)
        targets = [target.to(device) for target in list(source_targets + target_targets)]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % cfg.SOLVER.VAL_ITER == 0 and iteration != 0 :

                model.eval()
                val_results = validataion(cfg, model, val_data_loader[0], distributed=False)

                if type(val_results)== dict:
                    AP50_online = val_results['map'] * 100
                    meters.update(AP50=AP50_online)
                else:
                    val_results = val_results[0]
                    # used for saving model
                    AP50_online = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                    # used for logging
                    meter_AP50= val_results.results['bbox']['AP50'] * 100
                    meter_AP = val_results.results['bbox']['AP']* 100
                    meters.update(AP50=meter_AP50, AP =meter_AP )

                if AP50_online > max_AP50_bank:
                    max_AP50_bank = AP50_online
                    checkpointer.save("best_model_updated_{}_{:07d}".format(AP50_online, iteration), **arguments)

                model.train()

        if iteration == max_iter - 1:
            checkpointer.save("model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def validataion(cfg, model, data_loader, distributed=False):
    iou_types = ("bbox",)
    # iou_types = ("map",)
    dataset_name = cfg.DATASETS.TEST

    results = inference(
        model,
        data_loader,
        dataset_name=dataset_name,
        iou_types=iou_types,
        box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        output_folder=None,
    )
    # synchronize()
    return results