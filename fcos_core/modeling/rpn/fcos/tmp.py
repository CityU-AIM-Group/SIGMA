import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator, make_prototype_evaluator, PrototypeComputation
from fcos_core.layers import SigmoidFocalLoss, FocalLoss, CosineLoss, BCEFocalLoss
from fcos_core.layers import Scale

import matplotlib.pyplot as plt
import ipdb
import os
import numpy as np

eps = 1e-8
INF = 1e10


def see(data, name='default'):
    print('#################################', name, '#################################')
    print('max: ', torch.max(data))
    print('mean: ', torch.mean(data))
    print('min: ', torch.min(data))
    print('##########################################################################')


def save_feat(feat, name='./loco_sim10k/source.pt'):
    print(feat[0].size())
    feat = torch.cat(feat, dim=0).cpu()
    print(feat.size())
    torch.save(feat, name)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class GRAPHHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(GRAPHHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        if mode == 'in':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT

        middle_tower = []
        for i in range(num_convs):
            middle_tower.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if mode == 'in':
                middle_tower.append(nn.GroupNorm(32, in_channels))
            middle_tower.append(nn.ReLU())

        self.add_module('middle_tower', nn.Sequential(*middle_tower))

        # initialization
        for modules in [self.middle_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        middle_tower = []
        for l, feature in enumerate(x):
            middle_tower.append(self.middle_tower(feature))
        return middle_tower


class GRAPHModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(GRAPHModule, self).__init__()
        self.cfg = cfg.clone()
        # BASIC SETTINGS
        self.with_bg_proto = cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG
        self.with_bias_dc = cfg.MODEL.MIDDLE_HEAD.COND_WITH_BIAS
        self.with_concated_maps = cfg.MODEL.MIDDLE_HEAD.CAT_ACT_MAP
        self.with_shortcut_GCNs = cfg.MODEL.MIDDLE_HEAD.GCN_SHORTCUT
        self.with_global_gcn = cfg.MODEL.MIDDLE_HEAD.GLOBAL_GCN
        self.with_proto_alignment = cfg.MODEL.MIDDLE_HEAD.PROTO_ALIGN

        # CONFIG SETTINGS
        self.act_loss_cfg = cfg.MODEL.MIDDLE_HEAD.ACT_LOSS
        self.GCN_norm_cfg = cfg.MODEL.MIDDLE_HEAD.GCN_EDGE_NORM
        self.GCN_out_act_cfg = cfg.MODEL.MIDDLE_HEAD.GCN_OUT_ACTIVATION

        # HYPERPARAMETERS


        self.lamda1 = cfg.MODEL.MIDDLE_HEAD.GCN_LOSS_WEIGHT
        self.lamda2 = cfg.MODEL.MIDDLE_HEAD.ACT_LOSS_WEIGHT
        self.lamda3 = cfg.MODEL.MIDDLE_HEAD.CON_LOSS_WEIGHT
        self.lamda4 = cfg.MODEL.MIDDLE_HEAD.CON_LOSS_WEIGHT_TG


        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.num_classes_bg = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

        self.prototype_channel = cfg.MODEL.MIDDLE_HEAD.PROTO_CHANNEL
        head_in = GRAPHHead(cfg, in_channels, in_channels, mode='in')
        self.head_in = head_in
        self.consistency_tg_cfg = cfg.MODEL.MIDDLE_HEAD.CON_TG_CFG
        proto_cls_hidden_dim = 512
        cond_hidden_channel = 256
        self.relu = torch.nn.ReLU().to('cuda')

        # LOSS STTINGS
        self.node_loss_fn = nn.CrossEntropyLoss()
        self.cosine_KL_loss = CosineLoss()

        if self.with_bg_proto:
            self.proto_cls = torch.nn.Linear(proto_cls_hidden_dim, self.num_classes_bg).to('cuda')
            self.register_buffer('prototype', torch.zeros(self.num_classes_bg, self.prototype_channel))
            self.prototype_buffer_batch = torch.zeros(self.num_classes_bg, self.prototype_channel)
            if self.with_concated_maps:
                head_out = GRAPHHead(cfg, in_channels + self.num_classes_bg, in_channels, mode='out')
                self.head_out = head_out
            if self.act_loss_cfg == "softmaxFL":
                self.act_loss_func = FocalLoss(
                    self.num_classes_bg
                )
            elif self.act_loss_cfg == "sigmoidFL":
                self.act_loss_func = BCEFocalLoss()

            # self.act_loss_func = SigmoidFocalLoss(
            #     cfg.MODEL.FCOS.LOSS_GAMMA,
            #     cfg.MODEL.FCOS.LOSS_ALPHA
            # )
            # self.act_loss_func =  nn.BCEWithLogitsLoss()

        else:
            self.proto_cls = torch.nn.Linear(proto_cls_hidden_dim, self.num_classes).to('cuda')
            self.register_buffer('prototype', torch.zeros(self.num_classes, self.prototype_channel))
            self.prototype_buffer_batch = torch.zeros(self.num_classes, self.prototype_channel)
            if self.with_concated_maps:
                head_out = GRAPHHead(cfg, in_channels + self.num_classes, in_channels)
                self.head_out = head_out
            if self.with_act_loss:
                self.act_loss_func = SigmoidFocalLoss(
                    cfg.MODEL.FCOS.LOSS_GAMMA,
                    cfg.MODEL.FCOS.LOSS_ALPHA
                )
        # PROTOTYPE SETTINGS
        self.prototype_evaluator = make_prototype_evaluator(cfg)
        self.proto_cls_hidden = torch.nn.Linear(cfg.MODEL.MIDDLE_HEAD.GCN2_OUT_CHANNEL, proto_cls_hidden_dim).to(
            'cuda')
        self.momentum = cfg.MODEL.MIDDLE_HEAD.PROTO_MOMENTUM



        # GCNs SETTINFS
        self.edge_project_u = torch.nn.Linear(256, cfg.MODEL.MIDDLE_HEAD.GCN_EDGE_PROJECT).to('cuda')
        self.edge_project_v = torch.nn.Linear(256, cfg.MODEL.MIDDLE_HEAD.GCN_EDGE_PROJECT).to('cuda')
        self.gcn_layer1 = torch.nn.Linear(256, cfg.MODEL.MIDDLE_HEAD.GCN1_OUT_CHANNEL).to('cuda')
        self.gcn_layer2 = torch.nn.Linear(cfg.MODEL.MIDDLE_HEAD.GCN1_OUT_CHANNEL,
                                          cfg.MODEL.MIDDLE_HEAD.GCN2_OUT_CHANNEL).to('cuda')


        # CONDITIONAL CONV SETTINGS
        self.cond_1 = torch.nn.Linear(self.prototype_channel, cond_hidden_channel).to('cuda')
        if self.with_bias_dc:
            self.cond_2 = torch.nn.Linear(cond_hidden_channel, 257).to('cuda')
        else:
            self.cond_2 = torch.nn.Linear(cond_hidden_channel, 256).to('cuda')

        # initialization
        for i in [self.cond_1, self.cond_2,
                  self.gcn_layer1, self.gcn_layer2,
                  self.edge_project_u, self.edge_project_v,
                  self.proto_cls, self.proto_cls_hidden]:
            nn.init.normal_(i.weight, std=0.01)
            nn.init.constant_(i.bias, 0)



    def GCNs_global(self, x, Adj):
        # ATTENTION FORMAT GCNs

        x = self.relu(self.gcn_layer2(torch.mm(Adj, self.gcn_layer1(x))))
        if self.with_shortcut_GCNs:
            x += x
        return x

    def GCNs(self, nodes, Adj):
        x = nodes
        # layer 1
        x = self.relu(self.gcn_layer1(torch.mm(Adj, x)))
        # layer 2
        if self.GCN_out_act_cfg == 'softmax':
            x = (self.gcn_layer2(torch.mm(Adj, x))).softmax(dim=-1)
        elif self.GCN_out_act_cfg == 'sigmoid':
            x = (self.gcn_layer2(torch.mm(Adj, x))).sigmoid()
        elif self.GCN_out_act_cfg == 'tanh':
            x = (self.gcn_layer2(torch.mm(Adj, x))).tanh()
        elif self.GCN_out_act_cfg == 'relu':
            x = (self.relu(self.gcn_layer2(torch.mm(Adj, x))))
        elif self.GCN_out_act_cfg == 'NO':
            x = self.gcn_layer2(torch.mm(Adj, x))
        else:
            raise KeyError('unknown gcn output activation')

        if self.with_shortcut_GCNs:
            x = x + nodes
        return x

    def get_edge(self, nodes_feat):
        if self.GCN_norm_cfg == 'NO':
            Adj = torch.mm(nodes_feat, nodes_feat.t()).softmax(-1).detach()
            return Adj
        elif self.GCN_norm_cfg == 'softmax':
            Adj = torch.mm(self.edge_project_u(nodes_feat), self.edge_project_v(nodes_feat).t())
            return Adj.softmax(-1)
        elif self.GCN_norm_cfg == 'cosine_detached':
            Adj = sim_matrix(nodes_feat, nodes_feat).softmax(-1).detach()
            return Adj
        elif self.GCN_norm_cfg == 'cosine':
            # nodes_feat_pj = self.edge_project_v(self.relu(self.edge_project_u(nodes_feat)))
            nodes_feat_pj = self.relu(self.edge_project_v(nodes_feat))
            sim = sim_matrix(nodes_feat_pj, nodes_feat_pj)
            # Adj = sim.softmax(dim=-1)
            norm = torch.sum(sim, dim=-1)
            assert norm.min() > 0, '0 appears in norm'
            Adj = sim / torch.clamp(norm, min=eps)
            return Adj

    def forward(self, images, features, targets=None, return_maps=False):




        # for l, feature in enumerate(features):
        #     see(feature,'in')
        # os._exit(0)

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
        consistency_loss = 0
        features = self.head_in(features)
        num_classes_GCNs = self.num_classes_bg if self.with_bg_proto else self.num_classes
        if self.training and targets:
            # source: update prototype
            prototype_buffer_batch = features[0].new_zeros(self.prototype_buffer_batch.size())
            locations = self.compute_locations(features)
            pos_points, pos_labels, act_maps_labels = self.prototype_evaluator(
                locations, features, targets
            )
            # # ------------------Graph Reasoning----------------------
            if self.with_global_gcn:
                Adj = self.get_edge(pos_points)
                nodes_GCNs = self.GCNs_global(pos_points, Adj)

                for i in range(num_classes_GCNs):
                    indx = pos_labels == i if self.with_bg_proto else pos_labels == i + 1
                    if indx.any():
                        prototype_buffer_batch[i] = nodes_GCNs[indx].mean(dim=0)
                self.update_prototype(prototype_buffer_batch)

                logits = self.proto_cls(self.relu(self.proto_cls_hidden(nodes_GCNs)))
                target = (pos_labels).long() if self.with_bg_proto else (pos_labels - 1).long()
                node_loss = self.lamda1 * self.node_loss_fn(logits, target)
            else:
                label_indx = pos_labels.new_zeros((num_classes_GCNs))
                for i in range(num_classes_GCNs):
                    indx = pos_labels == i if self.with_bg_proto else pos_labels == i + 1
                    if indx.any():
                        label_indx[i] = 1
                        nodes = pos_points[indx]
                        Adj = self.get_edge(nodes)
                        test_nan(Adj)
                        # ------------------Graph Reasoning----------------------
                        nodes_GCNs = self.GCNs(nodes, Adj)
                        pos_points[indx] = nodes_GCNs
                        prototype_buffer_batch[i] = nodes_GCNs.mean(dim=0)

                # cosine, KL = self.cosine_KL_loss(prototype_buffer_batch, self.prototype, label_indx)
                self.update_prototype(prototype_buffer_batch)

                # test_nan(consistency_loss)
                logits = self.proto_cls(self.relu(self.proto_cls_hidden(pos_points)))
                target = (pos_labels).long() if self.with_bg_proto else (pos_labels - 1).long()
                node_loss = self.lamda1 * self.node_loss_fn(logits, target)
                test_nan(node_loss)
            conded_weight = self.cond_2(self.relu(self.cond_1(self.prototype.detach())))
            ##########################################################################
            ######################  train with activation loss #######################
            ##########################################################################
            if self.act_loss_cfg:
                act_maps_labels_flatten = []
                act_maps_preds_flatten = []
                return_act_maps = []
                for l, feature in enumerate(features):
                    act_maps_logits = self.dynamic_conv(feature, conded_weight, num_classes_GCNs)
                    act_maps = act_maps_logits.softmax(dim=1) if self.act_loss_cfg == 'softmaxFL' else act_maps_logits.sigmoid()
                    return_act_maps.append(act_maps)
                    if self.with_concated_maps:
                        features[l] = torch.cat([features[l], act_maps], dim=1)
                    act_maps_labels_flatten.append(act_maps_labels[l].reshape(-1))
                    act_maps_preds_flatten.append(act_maps_logits.permute(0, 2, 3, 1).reshape(-1, num_classes_GCNs))
                act_maps_preds_flatten = torch.cat(act_maps_preds_flatten, dim=0)
                act_maps_labels_flatten = torch.cat(act_maps_labels_flatten, dim=0)
                features = self.head_out(features) if self.with_concated_maps else features

                # Activation Map loss
                if self.act_loss_cfg == 'softmaxFL':
                    act_loss = self.lamda2 * self.act_loss_func(
                        act_maps_preds_flatten,
                        act_maps_labels_flatten.long()
                    )
                elif self.act_loss_cfg == 'sigmoidFL':
                    N = features[0].size(0)
                    num = len(act_maps_labels_flatten)
                    target_flatten = act_maps_labels_flatten.new_zeros((num, 2))
                    target_flatten[range(num), list(act_maps_labels_flatten)] = 1
                    act_loss = self.lamda2 * self.act_loss_func(
                        act_maps_preds_flatten,
                        target_flatten.float()
                    )
                else:
                    act_loss = None
                return features, (node_loss, consistency_loss), act_loss, return_act_maps
            else:

                ##########################################################################
                ####################  train without activation loss ######################
                ##########################################################################
                return_act_maps = []
                for l, feature in enumerate(features):
                    act_maps_logits = self.dynamic_conv(feature, conded_weight, num_classes_GCNs)
                    act_maps = act_maps_logits.softmax(
                        dim=1) if self.act_loss_cfg == 'softmaxFL' else act_maps_logits.sigmoid()
                    # act_maps = act_maps_logits.sigmoid()

                    return_act_maps.append(act_maps)

                    if self.cfg.MODEL.MIDDLE_HEAD.CAT_ACT_MAP:
                        features[l] = torch.cat([features[l], act_maps], dim=1)
                features = self.head_out(features) if self.with_concated_maps else features
                return features, (node_loss, consistency_loss), None, return_act_maps

        elif self.training and not targets and (self.prototype == 0).sum() < 256 and self.with_proto_alignment:

            prototype_buffer_batch = features[0].new_zeros(self.prototype_buffer_batch.size())
            conded_weight = self.cond_2(self.relu(self.cond_1(self.prototype.detach())))
            return_act_maps = []
            for l, feature in enumerate(features):
                # see(feature)
                act_maps_logits = self.dynamic_conv(feature, conded_weight, num_classes_GCNs)
                act_maps = act_maps_logits.softmax(
                    dim=1) if self.act_loss_cfg == 'softmaxFL' else act_maps_logits.sigmoid()
                return_act_maps.append(act_maps)

            pos_points, pos_labels, _ = self.prototype_evaluator(
                locations=None, features=features, targets=return_act_maps
            )
            # post processing
            if self.with_concated_maps:
                for l, feature in enumerate(features):
                    features[l] = torch.cat([features[l], return_act_maps[l]], dim=1)
                features = self.head_out(features) if self.with_concated_maps else features
            if pos_points is not None:
                label_indx = pos_labels.new_zeros((num_classes_GCNs))
                for i in range(num_classes_GCNs):
                    indx = pos_labels == i if self.with_bg_proto else pos_labels == i + 1
                    if indx.any():
                        label_indx[i] = 1
                        nodes = pos_points[indx]
                        Adj = self.get_edge(nodes)
                        test_nan(Adj)
                        # ------------------Graph Reasoning----------------------
                        nodes_GCNs = self.GCNs(nodes, Adj)
                        pos_points[indx] = nodes_GCNs
                        prototype_buffer_batch[i] = nodes_GCNs.mean(dim=0)

                cosine, KL = self.cosine_KL_loss(prototype_buffer_batch, self.prototype, label_indx)

                if self.consistency_tg_cfg == 'cosine':
                    consistency_loss = self.lamda4 * cosine
                elif self.consistency_tg_cfg == 'KLdiv':
                    consistency_loss = self.lamda4 * KL

                return features, (None, consistency_loss), None, return_act_maps
            else:
                return features, None, None, return_act_maps

        else:
            ##########################################################################
            #################### train on targets and inference ######################
            ##########################################################################
            # act_maps_logits = []
            return_act_maps = []
            num_classes_GCNs = self.num_classes_bg if self.with_bg_proto else self.num_classes
            conded_weight = self.cond_2(self.relu(self.cond_1(self.prototype.detach())))
            for l, feature in enumerate(features):
                act_maps_logits = self.dynamic_conv(feature, conded_weight, num_classes_GCNs)
                act_maps = act_maps_logits.softmax(
                    dim=1) if self.act_loss_cfg == 'softmaxFL' else act_maps_logits.sigmoid()
                return_act_maps.append(act_maps)
                if self.with_concated_maps:
                    features[l] = torch.cat([features[l], act_maps], dim=1)
                # ipdb.set_trace()
                if self.cfg.DEBUG.ACT_MAP:
                    debug_draw_maps(act_maps, l)

            if self.cfg.DEBUG.TSNE:
                debug_T_SNE(self.prototype, exit=False)

            features = self.head_out(features) if self.with_concated_maps else features
            return features, None, None, return_act_maps

    def update_prototype(self, prototype_batch, momentum=0.95):
        self.prototype = self.prototype * self.momentum + prototype_batch * (1- self.momentum)
        # self.prototype = self.prototype * (1 - momentum) + prototype_batch * momentum

    def dynamic_conv(self, features, kernel_par, num_classes):
        if self.with_bias_dc:
            # WITH BIAS TERM
            weight = kernel_par[:, :-1]
            bias = kernel_par[:, -1]
            weight = weight.view(num_classes, -1, 1, 1)
            return torch.nn.functional.conv2d(features, weight, bias=bias)
        else:
            weight = kernel_par.view(num_classes, -1, 1, 1)
            return torch.nn.functional.conv2d(features, weight)

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


def build_condgraph(cfg, in_channels):
    return GRAPHModule(cfg, in_channels)


def test_nan(para, name='gcn'):
    assert para.max() < INF, 'nan of {}'.format(name)
    return para


def debug_T_SNE(prototype, exit=False):
    from sklearn.manifold import TSNE
    root = '/home/wuyang/Pictures/visualization/tsne_prototype/'
    if not os.path.exists(root):
        os.mkdir(root)
    num_classes = prototype.size(0)
    TSNE_embedded = TSNE(n_components=2).fit_transform(prototype.cpu().numpy())
    legend = []
    legend_name = []

    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    cmap = get_cmap(num_classes)

    for i in range(num_classes):
        legend.append(plt.scatter(TSNE_embedded[i, 0], TSNE_embedded[i, 1], color=cmap(i), s=20))
        legend_name.append('class_{}'.format(i + 1))
    plt.legend(handles=legend, labels=legend_name, loc=1, prop={'size': 8})

    # plt.set_cmap('rainbow')
    plt.savefig(root + 'tsne.png', dpi=600)
    plt.close()
    if exit:
        os._exit(0)


def debug_draw_maps(act_maps, feat_level, exit=False):
    target_size = (1, 1, 800, 1333)
    root = '/home/wuyang/Pictures/visualization/activation_maps/'
    if not os.path.exists(root):
        os.mkdir(root)
    # ipdb.set_trace()
    for i in range(act_maps.size(0)):
        for cls, show_map in enumerate(act_maps[i]):
            show_map = show_map.view(1, 1, show_map.size(0), show_map.size(1))

            show_map = F.interpolate(show_map, size=(800, 1333), mode='bilinear').squeeze()
            show_map = show_map.cpu().numpy()
            print(np.mean(show_map))
            plt.figure()
            plt.imshow(show_map)

            plt.colorbar()
            plt.set_cmap('rainbow')

            plt.imsave(root + 'image-{}_level-{}_class-{}_map.png'.format(i, feat_level, cls), show_map)
            plt.close()

    if exit:
        os._exit(0)