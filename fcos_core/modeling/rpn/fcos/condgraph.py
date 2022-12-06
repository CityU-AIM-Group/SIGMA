# --------------------------------------------------------
# SCAN: Cross-domain Object Detection with Semantic Conditioned Adaptation 
# Written by Wuyang Li
# This file covers the core operation on the feature maps for domain adaptation
# https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/rpn/fcos/condgraph.py
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
from .loss import make_fcos_loss_evaluator, make_prototype_evaluator
from fcos_core.layers import FocalLoss, BCEFocalLoss
eps = 1e-8
INF = 1e10

        # POST PROCESSING
def see(data, name='default'):
    print('#################################', name, '#################################')
    print('max: ', torch.max(data))
    print('mean: ', torch.mean(data))
    print('min: ', torch.min(data))
    print('##########################################################################')

def sim_matrix(a, b, eps=eps):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class PROTOTYPECounter():
    def __init__(self,cycle=3, stop=False):
        self.cycle  = cycle
        self.counter = -1
        self.stop = stop
    def __call__(self, *args, **kwargs):
        # 0, 1, 2, 3, 4, ..., n, n, n, n
        if self.stop:
            if self.counter == self.cycle:
                return self.cycle
            else:
                self.counter += 1
                return  self.counter
        # 0, 1, 2, 3, 0, 1, 2, 3, ...
        else:
            self.counter += 1
            if self.counter == self.cycle:
                self.counter = 0
            return self.counter

class GRAPHHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(GRAPHHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        
        if mode == 'in':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = cfg.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

        middle_tower = []
        for i in range(num_convs):
            # if mode == 'in':
            #     middle_tower.append(nn.BatchNorm2d(in_channels))
                # if cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                #     middle_tower.append(nn.GroupNorm(32, in_channels))
                # elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                #     middle_tower.append(nn.InstanceNorm2d(in_channels))
                # elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                #     middle_tower.append(nn.BatchNorm2d(in_channels))
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
                if cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
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
    Module for Semantic Middle Head
    """

    def __init__(self, cfg, in_channels):
        super(GRAPHModule, self).__init__()

        # Debugger: for those developers
        self.debug_cfg = cfg.MODEL.DEBUG_CFG
        if self.debug_cfg:
            from fcos_core.vis_tools import VIS_TOOLS
            self.debugger = VIS_TOOLS()
        # from fcos_core.vis_tools import VIS_TOOLS
        # self.debugger = VIS_TOOLS()

        self.cfg = cfg.clone()

        # Basic settings
        self.with_bg_proto              = cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG
        self.with_bias_dc               = cfg.MODEL.MIDDLE_HEAD.COND_WITH_BIAS
        self.with_concated_maps         = cfg.MODEL.MIDDLE_HEAD.CAT_ACT_MAP
        self.with_shortcut_GCNs         = cfg.MODEL.MIDDLE_HEAD.GCN_SHORTCUT
        self.with_global_gcn            = cfg.MODEL.MIDDLE_HEAD.GLOBAL_GCN
        self.with_proto_uv              = cfg.MODEL.MIDDLE_HEAD.PROTO_MEAN_VAR
        self.with_self_training         = cfg.MODEL.MIDDLE_HEAD.GCN_SELF_TRAINING
        self.fpn_strides                = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_classes_fg             = cfg.MODEL.FCOS.NUM_CLASSES -1
        self.used_num_classes           = self.num_classes_fg + int(self.with_bg_proto)

        # Many choices
        self.transfer_cfg               = cfg.MODEL.MIDDLE_HEAD.TRANSFER_CFG # 'NODES', 'EDGE', 'ADJ'
        self.act_loss_cfg               = cfg.MODEL.MIDDLE_HEAD.ACT_LOSS
        self.GCN_norm_cfg               = cfg.MODEL.MIDDLE_HEAD.GCN_EDGE_NORM
        self.GCN_out_act_cfg            = cfg.MODEL.MIDDLE_HEAD.GCN_OUT_ACTIVATION
        self.tg_transfer_cfg            = cfg.MODEL.MIDDLE_HEAD.CON_TG_CFG
        
        # Hyperparameter
        self.lamda1                     = cfg.MODEL.MIDDLE_HEAD.GCN_LOSS_WEIGHT
        self.lamda2                     = cfg.MODEL.MIDDLE_HEAD.ACT_LOSS_WEIGHT
        self.lamda3                     = cfg.MODEL.MIDDLE_HEAD.CON_LOSS_WEIGHT
        self.lamda4                     = cfg.MODEL.MIDDLE_HEAD.GCN_LOSS_WEIGHT_TG


        # self.num_classes_fg_bg = cfg.MODEL.FCOS.NUM_CLASSES
        # Important settings
        self.use_rnn                    = cfg.MODEL.MIDDLE_HEAD.USE_RNN
        self.prototype_iter             = cfg.MODEL.MIDDLE_HEAD.PROTO_ITER # 1, 3, 9
        self.momentum                   = cfg.MODEL.MIDDLE_HEAD.PROTO_MOMENTUM
        self.relu                       = torch.nn.ReLU().to('cuda')
        prototype_channel               = cfg.MODEL.MIDDLE_HEAD.PROTO_CHANNEL
        cond_hidden_channel             = cfg.MODEL.MIDDLE_HEAD.COND_HIDDEN_CHANNEL
        proto_cls_hidden_dim            = 512

        # Pre-processing

        self.head_in = GRAPHHead(cfg, in_channels, in_channels, mode='in')

        # Prototype settings
        if self.prototype_iter ==1 :
            self.register_buffer('prototype', torch.randn(self.used_num_classes, prototype_channel))
        else:
            self.register_buffer('prototype', torch.randn(self.used_num_classes, prototype_channel, self.prototype_iter))

        self.prototype_buffer_batch     = torch.zeros(self.used_num_classes, prototype_channel)
        if self.with_proto_uv:
            self.register_buffer('prototype_var', torch.randn(self.used_num_classes, prototype_channel))
        if self.with_concated_maps:
            head_out                    = GRAPHHead(cfg, in_channels + self.used_num_classes, in_channels, mode='out')
            self.head_out = head_out
        if self.act_loss_cfg == "softmaxFL":
            self.act_loss_func = FocalLoss(
                self.used_num_classes
            )
        elif self.act_loss_cfg == "sigmoidFL":
            self.act_loss_func = BCEFocalLoss()

        self.prototype_evaluator        = make_prototype_evaluator(cfg)
        self.proto_cls_hidden           = torch.nn.Linear(cfg.MODEL.MIDDLE_HEAD.GCN2_OUT_CHANNEL, proto_cls_hidden_dim).to(
            'cuda')
        self.proto_cls                  = torch.nn.Linear(proto_cls_hidden_dim, self.num_classes_fg + int(self.with_bg_proto)).to('cuda')
        self.node_loss_fn               = nn.CrossEntropyLoss()


        # GCNs settings
        if self.with_global_gcn:
            self.multihead_attn         = MultiHeadAttention(256, 4, dropout=0.1)
        else:
            self.gcn_layer1             = torch.nn.Linear(256, cfg.MODEL.MIDDLE_HEAD.GCN1_OUT_CHANNEL).to('cuda')
            self.gcn_layer2             = torch.nn.Linear(cfg.MODEL.MIDDLE_HEAD.GCN1_OUT_CHANNEL,
                                              cfg.MODEL.MIDDLE_HEAD.GCN2_OUT_CHANNEL).to('cuda')
            # self.edge_project_u = torch.nn.Linear(256, cfg.MODEL.MIDDLE_HEAD.GCN_EDGE_PROJECT).to('cuda')
            # self.edge_project_v = torch.nn.Linear(256, cfg.MODEL.MIDDLE_HEAD.GCN_EDGE_PROJECT).to('cuda')
            for i in [
                      self.gcn_layer1, self.gcn_layer2,
                      # self.edge_project_u, self.edge_project_v,
                      ]:
                nn.init.normal_(i.weight, std=0.01)
                nn.init.constant_(i.bias, 0)


        # Dynamic Brunch
        if  self.use_rnn:
            print(self.use_rnn)
            self.cond_nx1 = torch.nn.Conv2d(512, 256,kernel_size=(self.prototype_iter, 1))

            if self.use_rnn == 'GRU':
                self.cond_rnn           = nn.GRU(256, 512, 2)
            elif self.use_rnn == 'LSTM':
                self.cond_rnn           = nn.LSTM(256, 512, 2)
            else:
                self.cond_rnn           = nn.RNN(256, 512, 2, nonlinearity='tanh')

            self.counter_rnn            = PROTOTYPECounter(self.prototype_iter, stop=True)
        elif self.prototype_iter > 1:
            self.counter                = PROTOTYPECounter(self.prototype_iter)
            self.cond_nx1               = torch.nn.Conv2d(prototype_channel, cond_hidden_channel, kernel_size=(self.prototype_iter,1))
            nn.init.normal_( self.cond_nx1.weight)
            nn.init.constant_( self.cond_nx1.bias, 0)
            self.cond_nx1_norm          = torch.nn.GroupNorm(32, cond_hidden_channel)
        elif self.prototype_iter == 1:
            self.cond_1                 = torch.nn.Linear(prototype_channel, cond_hidden_channel).to('cuda')
            nn.init.normal_( self.cond_1.weight, std=0.01)
            nn.init.constant_( self.cond_1.bias, 0)
        self.cond_2                     = torch.nn.Linear(cond_hidden_channel, 256 + int(self.with_bias_dc)).to('cuda')

        # Semantic transfer settings

        if ('ADJ' in self.transfer_cfg) or ('ADJ_COMPLETE' in self.transfer_cfg):
            self.transfer_loss_inter_class      = nn.CosineEmbeddingLoss(margin=0.0)
        if ('NODES' in self.transfer_cfg) or 'PROTOTYPE' in self.transfer_cfg :
            self.transfer_loss_prototype        = nn.KLDivLoss()

        # self.transfer_loss = SinkhornDistance(eps=1, max_iter=100, reduction='mean')
        # Initialization
        for i in [self.cond_2,
                  # self.gcn_layer1, self.gcn_layer2,
                  # self.edge_project_u, self.edge_project_v,
                  self.proto_cls, self.proto_cls_hidden]:
            nn.init.normal_(i.weight, std=0.01)
            nn.init.constant_(i.bias, 0)


    def GCNs_global(self, x, Adj):
        # transformer
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

    def update_prototype_ensemble(self, prototype_buffer_batch):
        # print(self.prototype[:,0])
        # print(self.prototype[:,0,0])
        # print(self.prototype[:,0,1])
        # print(self.prototype[:,0,2])
        # print('----------------------------')
        if self.use_rnn:
            self.update_prototype_nx1_rnn(prototype_buffer_batch)
        elif self.prototype_iter > 1:
            self.update_prototype_nx1(prototype_buffer_batch)
        else:
            self.update_prototype(prototype_buffer_batch)

    def get_conded_weight(self):
        if self.use_rnn:
            # num_classes_fg_bg, channel, iter [9, 256, 3]
            conded_weight = self.cond_nx1(
                self.cond_rnn(
                    self.prototype.permute(2, 0, 1))[0].permute(1, 2, 0).unsqueeze(-1)
            ).squeeze()
        elif self.prototype_iter > 1:
            conded_weight = self.cond_2(
                self.relu(self.cond_nx1_norm(
                    self.cond_nx1(
                        self.prototype.unsqueeze(-1)
                    )
                ).squeeze())
            )
        # original setting: num_classes_fg_bg, channel, [9, 256]
        else:
            conded_weight = self.cond_2(
                self.relu(
                    self.cond_1(self.prototype)
                )
            )

        return conded_weight

    def get_act_loss(self, features, conded_weight,act_maps_labels):
        act_maps_labels_flatten = []
        act_maps_preds_flatten = []
        act_maps_return = []
        for l, feature in enumerate(features):
            act_maps_logits = self.dynamic_conv(feature, conded_weight)
            act_maps = act_maps_logits.softmax(
                dim=1) if self.act_loss_cfg == 'softmaxFL' \
                else act_maps_logits.sigmoid()
            act_maps_return.append(act_maps)
            act_maps_labels_flatten.append(act_maps_labels[l].reshape(-1))
            act_maps_preds_flatten.append(act_maps_logits.permute(0, 2, 3, 1).reshape(-1, self.used_num_classes))
        act_maps_preds_flatten = torch.cat(act_maps_preds_flatten, dim=0)
        act_maps_labels_flatten = torch.cat(act_maps_labels_flatten, dim=0)

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
        return act_loss, act_maps_return

    def GCNs_post_processing(self,nodes_GCNs,pos_points):
        if self.with_shortcut_GCNs:
            nodes_GCNs = (nodes_GCNs + pos_points).squeeze()
        else:
            nodes_GCNs = nodes_GCNs.squeeze()
        return nodes_GCNs

    def features_post_processing(self, features, act_maps):
        if self.with_concated_maps:
            for l, feature in enumerate(features):
                features[l] = torch.cat([features[l], act_maps[l]], dim=1)
            features = self.head_out(features)
        return features

    def _forward_gcns(self, pos_points, pos_labels):
        prototype_buffer_batch = pos_points[0].new_zeros(self.prototype_buffer_batch.size())
        if self.with_global_gcn:
            # batch = 0
            pos_points = pos_points.unsqueeze(0)

            nodes_GCNs = self.multihead_attn(pos_points, pos_points, pos_points)[0]
            nodes_GCNs = self.GCNs_post_processing(nodes_GCNs, pos_points)

            for i in range(self.used_num_classes):
                indx = pos_labels == i if self.with_bg_proto else pos_labels == i + 1
                if indx.any():
                    prototype_buffer_batch[i] = nodes_GCNs[indx].mean(dim=0)

            logits = self.proto_cls(self.relu(self.proto_cls_hidden(nodes_GCNs)))
            target = (pos_labels).long() if self.with_bg_proto else (pos_labels - 1).long()
            node_loss = self.lamda1 * self.node_loss_fn(logits, target)
        else:
            label_indx = pos_labels.new_zeros((self.used_num_classes))
            for i in range(self.used_num_classes):
                indx = pos_labels == i if self.with_bg_proto else pos_labels == i + 1
                if indx.any():
                    label_indx[i] = 1
                    nodes = pos_points[indx]
                    Adj = self.get_edge(nodes)
                    test_nan(Adj)
                    nodes_GCNs = self.GCNs(nodes, Adj)
                    pos_points[indx] = nodes_GCNs
                    prototype_buffer_batch[i] = nodes_GCNs.mean(dim=0)

            logits = self.proto_cls(self.relu(self.proto_cls_hidden(pos_points)))
            target = (pos_labels).long() if self.with_bg_proto else (pos_labels - 1).long()
            node_loss = self.lamda1 * self.node_loss_fn(logits, target)

        test_nan(node_loss)
        return node_loss, prototype_buffer_batch

    def _forward_train_source(self, features, targets=None, return_maps=False):
        transfer_loss = 0
        node_loss = 0
        act_loss = 0
        
        locations = self.compute_locations(features)
        pos_points, pos_labels, act_maps_labels = self.prototype_evaluator(
            locations, features, targets
        )
        node_loss, prototype_batch = self._forward_gcns(pos_points,pos_labels)
        self.update_prototype_ensemble(prototype_batch)
        # obtain prototype conditioned weight
        conded_weight = self.get_conded_weight()
        if self.act_loss_cfg:
            act_loss, act_maps_return =  self.get_act_loss(features,conded_weight, act_maps_labels)
            features = self.features_post_processing(features, act_maps_return)
            return features, node_loss, transfer_loss, act_loss, act_maps_return
        else:
            act_maps_return = []
            for l, feature in enumerate(features):
                act_maps_logits = self.dynamic_conv(feature, conded_weight)
                act_maps = act_maps_logits.softmax(
                    dim=1) if self.act_loss_cfg == 'softmaxFL' else act_maps_logits.sigmoid()
                act_maps_return.append(act_maps)

            features = self.features_post_processing(features, act_maps_return)
            return features, node_loss, 0, None, act_maps_return
    def get_transfer_loss(self, tg_prototype, tg_nodes, tg_labels):



        losses = {}
        sr_prototype = self.prototype.mean(dim=-1).detach() if self.prototype_iter > 1 \
            else self.prototype.detach()

        if 'NODES' in self.transfer_cfg:
            transfer_loss_node =  self.transfer_loss_prototype(tg_nodes.softmax(-1).log(), sr_prototype[tg_labels.long()].softmax(-1))
            transfer_loss_node = {"trans_proto_node": transfer_loss_node}
            losses.update(transfer_loss_node)

        if 'PROTOTYPE' in self.transfer_cfg:

            indx = tg_prototype.sum(-1).bool()
            transfer_loss_prototype =  self.transfer_loss_prototype(tg_prototype[indx].softmax(-1).log(), sr_prototype[indx].softmax(-1))
            transfer_loss_prototype = {"transfer_loss_prototype": transfer_loss_prototype}
            losses.update(transfer_loss_prototype)

        if 'ADJ' in self.transfer_cfg:
            indx = tg_prototype.sum(dim=-1).bool()
            adj_sr = sim_matrix(sr_prototype[indx], sr_prototype[indx]).view(1,-1)
            adj_tg = sim_matrix(tg_prototype[indx], tg_prototype[indx]).view(1,-1)
            cosine_target = adj_sr.new_ones(adj_sr.size())
            transfer_loss_inter_class = self.transfer_loss_inter_class(adj_sr,adj_tg,cosine_target)
            transfer_loss_inter_class = {"adj_loss": transfer_loss_inter_class}
            losses.update(transfer_loss_inter_class)
        if 'ADJ_COMPLETE' in self.transfer_cfg:
            _indx = ~(tg_prototype.sum(dim=-1).bool())
            tg_prototype_complete= tg_prototype
            tg_prototype_complete[_indx] = sr_prototype[_indx]
            adj_sr = sim_matrix(sr_prototype, sr_prototype).view(1,-1)
            adj_tg = sim_matrix(tg_prototype_complete, tg_prototype_complete).view(1,-1)
            cosine_target = adj_sr.new_ones(adj_sr.size())
            transfer_loss_inter_class_complete = self.transfer_loss_inter_class(adj_sr,adj_tg,cosine_target)
            transfer_loss_inter_class_complete = {"adj_complete_loss": transfer_loss_inter_class_complete}
            losses.update(transfer_loss_inter_class_complete)

        if len(losses) > 0:
            transfer_loss = sum(loss for loss in losses.values())
            return transfer_loss
        else:
            return None

    def _forward_train_target(self, features, targets=None, return_maps=False, forward_target=False):

        transfer_loss = 0
        node_loss = 0
        act_loss = 0


        act_maps_return = []
        for l, feature in enumerate(features):
            conded_weight = self.get_conded_weight()
            act_maps_logits = self.dynamic_conv(feature, conded_weight)
            act_maps = act_maps_logits.softmax(
                dim=1) if self.act_loss_cfg == 'softmaxFL' else act_maps_logits.sigmoid()
            act_maps_return.append(act_maps)
        # Post-processing
        features = self.features_post_processing(features, act_maps_return)
        if forward_target:
            pos_points, pos_labels, _ = self.prototype_evaluator(
                locations=None, features=features, targets=act_maps_return
            )
            # sampled nodes exist
            if (pos_points is not None) and ((self.transfer_cfg[0] is not None ) or (self.with_self_training is True)):
                # node_loss can be used for self-training
                node_loss, tg_prototype_batch = self._forward_gcns(pos_points, pos_labels)
                node_loss = self.lamda4 * node_loss * int(self.with_self_training)
                transfer_loss = self.get_transfer_loss(tg_prototype_batch, pos_points, pos_labels)


        return features, node_loss, transfer_loss, act_loss, act_maps_return


    def _forward_inference(self, features, targets=None, return_maps=False):
        act_maps_return = []
        conded_weight = self.get_conded_weight()
        for l, feature in enumerate(features):
            act_maps_logits = self.dynamic_conv(feature, conded_weight)
            act_maps = act_maps_logits.softmax(
                dim=1) if self.act_loss_cfg == 'softmaxFL' else act_maps_logits.sigmoid()
            act_maps_return.append(act_maps)
            if self.debug_cfg and 'ACT_MAP' in self.debug_cfg:
                self.debugger.debug_draw_maps(act_maps, l,name='activation_maps', exit=False)
        if self.debug_cfg and 'TSNE' in self.debug_cfg:
            self.debugger.debug_T_SNE(self.prototype, name='tsne_prototype', exit=False)
        if self.debug_cfg and 'GT' in self.debug_cfg:
            self.loss_evaluator = make_fcos_loss_evaluator(self.cfg)
            locations = self.compute_locations(features)
            box_cls_gt, box_reg_gt, box_ctr_gt = self.loss_evaluator.replace_targets(
                locations, act_maps_return, None, None, targets
            )
            for lvl, gt in enumerate(box_cls_gt):
                self.debugger.debug_draw_maps(gt, lvl, name='groundtruth', exit=False)
        if self.debug_cfg and 'CLEAN_ACT_MAP' in self.debug_cfg:
            from fcos_core.layers import MeanShift_GPU
            meanshift = MeanShift_GPU(batch_size=10000, bandwidth=0.1)
            for lvl, act_map in enumerate(act_maps_return):
                N, C, H, W = act_map.size()
                batch_clean_maps= []

                for img in range(N):
                    img_clean_maps = []
                    for category in range(C):
                        sampled_img = act_map[img,category,:,: ]
                        labels, centers = meanshift.fit(sampled_img.view(-1, 1))
                        clean_map = labels.view(sampled_img.size()).unsqueeze(0)
                        img_clean_maps.append(clean_map)
                    img_clean_maps = torch.cat(img_clean_maps,dim=0).unsqueeze(0)
                    batch_clean_maps.append(img_clean_maps)
                batch_clean_maps = torch.cat(batch_clean_maps,dim=0)
                self.debugger.debug_draw_maps(batch_clean_maps, lvl, name='clean_activation_maps_new_10000', exit=False)
                os._exit(0)

        features = self.features_post_processing(features, act_maps_return)
        return features,  act_maps_return

    def forward(self, features, targets=None, return_maps=False, mode='source', forward_target=False):
        if self.training:

            features_s, features_t = features
            features_s = self.head_in(features_s)
            features_t = self.head_in(features_t)
            features_s, node_loss_s, transfer_loss_s, act_loss_s, act_maps_s= self._forward_train_source(features_s, targets, return_maps)
            features_t, node_loss_t, transfer_loss_t, act_loss_t, act_maps_t = self._forward_train_target(features_t, targets=None, return_maps=return_maps,forward_target=forward_target)
            loss_dict = {
                "node_loss_s":  node_loss_s,
                "node_loss_t":  node_loss_t,
                "transfer_loss_s": transfer_loss_s,
                "transfer_loss_t": transfer_loss_t,
                "act_loss": act_loss_s,


            }
            return (features_s,features_t), loss_dict, (act_maps_s,act_maps_t)

        else:
            features = self.head_in(features)
            # features, _, _,act_maps = self._forward_inference(features, targets=None, return_maps=return_maps)
            return self._forward_inference(features, targets=None, return_maps=return_maps)

    def update_prototype(self, prototype_batch, momentum=0.95, mode='mean'):

        exist_indx = prototype_batch.sum(-1).bool()
        prototype_batch = prototype_batch.detach()

        if self.cfg.MODEL.MIDDLE_HEAD.COSINE_UPDATE_ON:
            momentum = torch.cosine_similarity(self.prototype[exist_indx], prototype_batch[exist_indx]).unsqueeze(1)
            self.prototype[exist_indx] = self.prototype[exist_indx] * momentum + prototype_batch[exist_indx] * (1 - momentum)
        else:
            self.prototype[exist_indx] = self.prototype[exist_indx] * momentum + prototype_batch[exist_indx] * (1 - momentum)
    def update_prototype_nx1(self, prototype_batch, momentum=0.95):
        iter = self.counter()
        exist_indx = prototype_batch.sum(-1).bool()
        prototype_batch = prototype_batch.detach()

        if self.cfg.MODEL.MIDDLE_HEAD.COSINE_UPDATE_ON:
            momentum = torch.cosine_similarity(self.prototype[exist_indx, :, iter] , prototype_batch[exist_indx]).unsqueeze(1)
            # print(momentum)
            self.prototype[exist_indx, :, iter] = self.prototype[exist_indx, :, iter] * momentum + \
                                                  prototype_batch[exist_indx]* (1 - momentum)
        else:
            self.prototype[exist_indx, :, iter] = self.prototype[exist_indx, :, iter].detach() * momentum + \
                                                  prototype_batch[exist_indx] * (1 - momentum)
    def update_prototype_nx1_rnn(self, prototype_batch, momentum=0.95):

        iter = self.counter_rnn()
        exist_indx = prototype_batch.sum(-1).bool()
        prototype_batch = prototype_batch.detach()

        if self.cfg.MODEL.MIDDLE_HEAD.COSINE_UPDATE_ON:
            if iter == self.prototype_iter:
                momentum = torch.cosine_similarity(self.prototype[exist_indx, :, iter-1], prototype_batch[exist_indx]).unsqueeze(1)
                # move
                for i in range(iter-1):
                    self.prototype[:, :, i] = self.prototype[:, :, i+1]
                # update t+1
                self.prototype[exist_indx, :, iter-1] = self.prototype[exist_indx, :, iter-1]* momentum \
                                                        + prototype_batch[exist_indx] * (1 - momentum)
            else:
                momentum = torch.cosine_similarity(self.prototype[exist_indx, :, iter], prototype_batch[exist_indx]).unsqueeze(1)
                self.prototype[exist_indx, :, iter] = self.prototype[exist_indx, :, iter] * momentum \
                                             + prototype_batch[exist_indx] * (1 - momentum)
        else:
            if iter == self.prototype_iter:
                for i in range(iter - 1):
                    self.prototype[:, :, i] = self.prototype[:, :, i + 1]
                # update t+1
                self.prototype[exist_indx, :, iter-1] = self.prototype[exist_indx, :, iter-1] * momentum + \
                                                          prototype_batch[exist_indx] * (
                                                                  1 - momentum)
            else:
                self.prototype[exist_indx, :, iter] = self.prototype[exist_indx, :, iter] * momentum \
                                             + prototype_batch[exist_indx] * (1 - momentum)


    def dynamic_conv(self, features, kernel_par):
        num_classes = self.used_num_classes
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

    def save_targets(self, cfg, features, act_maps_return, conded_weight, targets):
        # Save many features
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)
        locations = self.compute_locations(features)
        box_cls_gt, box_reg_gt, box_ctr_gt = self.loss_evaluator.replace_targets(
            locations, act_maps_return, None, None, targets
        )
        self.debugger.save_feat(box_cls_gt, id='target_gt')
        self.debugger.save_feat(act_maps_return, id='target_act_maps')
        self.debugger.save_feat(features,id='target_feats')
        self.debugger.save_feat(conded_weight,id='cond_weitht')
        self.debugger.save_feat(self.prototype,id='prototype')
        os._exit(0)



def build_condgraph(cfg, in_channels):
    return GRAPHModule(cfg, in_channels)


def test_nan(para, name='gcn'):
    assert para.max() < INF, 'nan of {}'.format(name)
    return para


