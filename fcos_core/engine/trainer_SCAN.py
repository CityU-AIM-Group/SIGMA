# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import ipdb
import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

from fcos_core.structures.image_list import to_image_list
import os
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.utils.miscellaneous import mkdir
from .validation import _inference
from fcos_core.utils.comm import synchronize
from fcos_core.structures.image_list import ImageList

#
def foward_detector(cfg, model, images, targets=None, return_maps=True,  DA_ON=True):
    with_middle_head = cfg.MODEL.MIDDLE_HEAD.CONDGRAPH_ON
    with_rcnn = not cfg.MODEL.RPN_ONLY


    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()
    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    if model_fcos.training and DA_ON:
        losses = {}
        images_s, images_t = images
        features_s = model_backbone(images_s.tensors)
        features_t = model_backbone(images_t.tensors)

        if with_middle_head:
            model_middle_head = model["middle_head"]
            (features_s, features_t), middle_head_losses,  (act_maps_s, act_maps_t) = model_middle_head((features_s,features_t), targets=targets )
            losses.update(middle_head_losses)
        else:
            act_maps = None

        if with_rcnn:
            proposals, proposal_losses = model_fcos(
                images_s, features_s, targets=targets)
            model_roi_head = model["roi_head"]
            feats, proposals, roi_head_loss = model_roi_head(features_s, proposals, targets)
            losses.update(proposal_losses)
            losses.update(roi_head_loss)
            f_s = {
                "P3": features_s[0]
            }
            f_t = {
                "P3": features_s[0]
            }
            return losses, (f_s, f_t)
        else:

            proposals, proposal_losses, score_maps = model_fcos(
                images_s, features_s, targets=targets)

            f_s = {
                layer: features_s[map_layer_to_index[layer]]
                for layer in feature_layers
            }
            f_t = {
                layer: features_t[map_layer_to_index[layer]]
                for layer in feature_layers
            }

            a_s = {
                layer: act_maps_s[map_layer_to_index[layer]]
                for layer in feature_layers
            }
            a_t = {
                layer: act_maps_t[map_layer_to_index[layer]]
                for layer in feature_layers
            }

            losses.update(proposal_losses)

            return losses, (f_s, f_t), (a_s, a_t)

    elif model_fcos.training  and not DA_ON:
        losses = {}
        features = model_backbone(images.tensors)

        # if with_middle_head:
        #     model_middle_head = model["middle_head"]
        #     matching_loss = model_middle_head((images_s, images_t),(features_s,features_t), targets=targets )

        if with_rcnn:
            proposals, proposal_losses = model_fcos(
                images, features, targets=targets)
            model_roi_head = model["roi_head"]
            # ipdb.set_trace()
            feats, proposals, roi_head_loss = model_roi_head(features, proposals, targets)
            losses.update(proposal_losses)
            losses.update(roi_head_loss)
            # print(losses)
            return losses, []
        else:
            proposals, proposal_losses, score_maps = model_fcos(
                images, features, targets=targets)
            losses.update(proposal_losses)
            return losses, []

    else:

        images = to_image_list(images)
        features = model_backbone(images.tensors)

        if with_middle_head:
            model_middle_head = model["middle_head"]
            features, act_maps = model_middle_head(features, targets=targets )
        if with_rcnn:
            model_roi_head = model["roi_head"]
            proposals, proposal_losses = model_fcos(
                images, features, targets=targets)
            feats, proposals, roi_head_loss = model_roi_head(features, proposals, targets)
        else:
            proposals, proposal_losses, score_maps = model_fcos(
                images, features, targets=targets, return_maps=return_maps)

        return proposals


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

def validataion(cfg, model, data_loader, distributed=False):
    if distributed:
        model["backbone"] = model["backbone"].module
        model["fcos"] = model["fcos"].module
    iou_types = ("bbox",)
    dataset_name = cfg.DATASETS.TEST
    assert len(data_loader) == 1, "More than one validation sets!"
    data_loader = data_loader[0]
    # for  dataset_name, data_loader_val in zip( dataset_names, data_loader):
    results, _ = _inference(
        cfg,
        model,
        data_loader,
        dataset_name=dataset_name,
        iou_types=iou_types,
        box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        output_folder=None,
    )
    synchronize()
    return results

def do_train(
        model,
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
):
    with_DA = cfg.MODEL.DA_ON
    data_loader_source = data_loader["source"]
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")

    for k in model:
        model[k].train()
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    AP50 = cfg.SOLVER.INITIAL_AP50
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    print('DA_ON: {}'.format(str(with_DA)))

    if not with_DA:
        max_iter = len(data_loader_source)
        for iteration, (images_s,targets_s, _) in enumerate(data_loader_source, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration
            if not pytorch_1_1_0_or_later:
                for k in scheduler:
                    scheduler[k].step()
            images_s = images_s.to(device)
            targets_s = [target_s.to(device) for target_s in targets_s]
            for k in optimizer:
                optimizer[k].zero_grad()

            loss_dict, features_s = foward_detector(cfg, model, images_s, targets=targets_s, DA_ON=False)
            loss_dict = {k + "_gs": loss_dict[k] for k in loss_dict}
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_gs=losses_reduced, **loss_dict_reduced)
            losses.backward()

            for k in optimizer:
                optimizer[k].step()
            if pytorch_1_1_0_or_later:

                for k in scheduler:
                    scheduler[k].step()
            # End of training
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr_backbone: {lr_backbone:.6f}",
                        "lr_fcos: {lr_fcos:.6f}",
                        "max mem: {memory:.0f}",
                    ]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                        lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0, ))
            if cfg.SOLVER.ADAPT_VAL_ON:
                if iteration % cfg.SOLVER.VAL_ITER == 0:
                    val_results = validataion(cfg, model, data_loader["val"], distributed)
                    # used for saving model
                    AP50_emp = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                    # used for logging
                    meter_AP50= val_results.results['bbox']['AP50'] * 100
                    meter_AP = val_results.results['bbox']['AP']* 100
                    meters.update(AP = meter_AP, AP50 = meter_AP50 )

                    if AP50_emp > AP50:
                        AP50 = AP50_emp
                        checkpointer.save("model_{}_{:07d}".format(AP50, iteration), **arguments)
                        print('***warning****,\n best model updated. {}: {}, iter: {}'.format(cfg.SOLVER.VAL_TYPE, AP50,
                                                                                           iteration))
                    if distributed:
                        model["backbone"] = model["backbone"].module
                        model["middle_head"] = model["middle_head"].module
                        model["fcos"] = model["fcos"].module
                    for k in model:
                        model[k].train()
            else:
                if iteration % checkpoint_period == 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)
            # save the last model
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)
    else:

        data_loader_target = data_loader["target"]
        max_iter = max(len(data_loader_source), len(data_loader_target))
        ga_dis_lambda = arguments["ga_dis_lambda"]
        used_feature_layers = arguments["use_feature_layers"]

        assert len(data_loader_source) == len(data_loader_target)
        for iteration, ((images_s, targets_s, _), (images_t, targets_t, _)) in enumerate(zip(data_loader_source, data_loader_target), start_iter):

            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration
            if not pytorch_1_1_0_or_later:
                for k in scheduler:
                    scheduler[k].step()

            images_s = images_s.to(device)
            images_t = images_t.to(device)
            targets_s = [target_s.to(device) for target_s in targets_s]

            for k in optimizer:
                optimizer[k].zero_grad()

            loss_dict, features_s_t, act_maps = foward_detector(cfg,
                model, (images_s, images_t), targets=targets_s, return_maps=True)
            # loss_dict = {k + "_gs": loss_dict[k] for k in loss_dict}

            for layer in used_feature_layers:
                features_s, features_t = features_s_t
                act_maps_s, act_maps_t = act_maps
                loss_dict["loss_adv_%s" % layer] = \
                    ga_dis_lambda * model["dis_%s_CON" % layer]((features_s[layer],features_t[layer]), act_maps=(act_maps_s[layer], act_maps_t[layer]))


            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_ds=losses_reduced, **loss_dict_reduced)

            losses.backward()
            del loss_dict, losses

            for k in optimizer:
                optimizer[k].step()

            if pytorch_1_1_0_or_later:
                for k in scheduler:
                    scheduler[k].step()

            # End of training
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            sample_layer = used_feature_layers[0]  # sample any one of used feature layer
            sample_optimizer = optimizer["dis_%s" % sample_layer]

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr_backbone: {lr_backbone:.6f}",
                        "lr_middle_head: {lr_middle_head:.6f}",
                        "lr_fcos: {lr_fcos:.6f}",
                        "lr_dis: {lr_dis:.6f}",
                        "max mem: {memory:.0f}",
                    ]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                        lr_middle_head=optimizer["middle_head"].param_groups[0]["lr"],
                        lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                        lr_dis=sample_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    ))

            if cfg.SOLVER.ADAPT_VAL_ON:
                if iteration % cfg.SOLVER.VAL_ITER== 0:
                    val_results = validataion(cfg, model, data_loader["val"], distributed)
                    # used for saving model
                    AP50_emp = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                    # used for logging
                    meter_AP50 = val_results.results['bbox']['AP50'] * 100
                    meter_AP = val_results.results['bbox']['AP'] * 100
                    meters.update(AP=meter_AP, AP50=meter_AP50)
                    if AP50_emp > AP50:
                        AP50 = AP50_emp
                        checkpointer.save("model_{}_{:07d}".format(AP50, iteration), **arguments)
                        print('***warning****,\n best model updated. {}: {}, iter: {}'.format(cfg.SOLVER.VAL_TYPE, AP50, iteration))
                    if distributed:
                        model["backbone"] = model["backbone"].module
                        model["middle_head"] = model["middle_head"].module
                        model["fcos"] = model["fcos"].module
                    for k in model:
                        model[k].train()
            else:
                if iteration % checkpoint_period == 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)))
