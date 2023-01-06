# --------------------------------------------------------
# SIGMA++: Improved Semantic-complete Graph Matching for Domain Adaptive Object Detection
# Written by Wuyang Li
# Based on https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/rpn/fcos/condgraph.py
# --------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from .loss import make_prototype_evaluator
from fcos_core.layers import BCEFocalLoss, MultiHeadAttention, Affinity, HyperGraph
import sklearn.cluster as cluster
from fcos_core.modeling.discriminator.layer import GradientReversal
import logging

class V2GConv(torch.nn.Module):
    # Project the sampled visual features to the graph embeddings:
    # visual features: [0,+INF) -> graph embedding: (-INF, +INF)
    def __init__(self, cfg, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(GRAPHHead, self).__init__()
        if mode == 'in':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = cfg.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

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
                if cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
            if i != (num_convs - 1):
                middle_tower.append(nn.ReLU())

        self.add_module('middle_tower', nn.Sequential(*middle_tower))

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

def build_V2G_linear(cfg):
    if cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN == 2:
        head_in_ln = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False),
        )
    elif cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN == 1:
        head_in_ln = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False),
        )
    elif cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN == 0:
        head_in_ln = nn.LayerNorm(256, elementwise_affine=False)
    else:
        head_in_ln = nn.LayerNorm(256, elementwise_affine=True)
    return head_in_ln


class GModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(GModule, self).__init__()

        init_item = []
        self.cfg = cfg.clone()
        self.logger = logging.getLogger("fcos_core.trainer")
        self.logger.info('node dis setting: ' + str(cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_PLACE))
        self.logger.info('use hyper graph: ' + str(cfg.MODEL.MIDDLE_HEAD.GM.WITH_HyperGNN))
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES

        self.with_hyper_graph = cfg.MODEL.MIDDLE_HEAD.GM.WITH_HyperGNN
        self.num_hyper_edge = cfg.MODEL.MIDDLE_HEAD.GM.HyperEdgeNum
        self.num_hypergnn_layer = cfg.MODEL.MIDDLE_HEAD.GM.NUM_HYPERGNN_LAYER
        self.angle_eps = cfg.MODEL.MIDDLE_HEAD.GM.ANGLE_EPS

        # One-to-one (o2o) matching or many-to-many (m2m) matching?
        self.matching_cfg = cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_CFG  # 'o2o' and 'm2m'
        self.with_cluster_update = cfg.MODEL.MIDDLE_HEAD.GM.WITH_CLUSTER_UPDATE  # add spectral clustering to update seeds
        self.with_semantic_completion = cfg.MODEL.MIDDLE_HEAD.GM.WITH_SEMANTIC_COMPLETION  # generate hallucination nodes

        # add quadratic matching constraints.
        self.with_quadratic_matching = cfg.MODEL.MIDDLE_HEAD.GM.WITH_QUADRATIC_MATCHING

        # Several weights hyper-parameters
        self.weight_matching = cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_WEIGHT
        self.weight_nodes = cfg.MODEL.MIDDLE_HEAD.GM.NODE_LOSS_WEIGHT
        self.weight_dis = cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_WEIGHT
        self.lambda_dis = cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_LAMBDA

        # Detailed settings
        self.with_domain_interaction = cfg.MODEL.MIDDLE_HEAD.GM.WITH_DOMAIN_INTERACTION
        self.with_complete_graph = cfg.MODEL.MIDDLE_HEAD.GM.WITH_COMPLETE_GRAPH
        self.with_node_dis = cfg.MODEL.MIDDLE_HEAD.GM.WITH_NODE_DIS
        self.with_global_graph = cfg.MODEL.MIDDLE_HEAD.GM.WITH_GLOBAL_GRAPH

        # Test 3 positions to put the node alignment discriminator. (the former is better)
        self.node_dis_place = cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_PLACE

        # future work
        self.with_cond_cls = cfg.MODEL.MIDDLE_HEAD.GM.WITH_COND_CLS  # use conditional kernel for node classification? (didn't use)
        self.with_score_weight = cfg.MODEL.MIDDLE_HEAD.GM.WITH_SCORE_WEIGHT  # use scores for node loss (didn't use)

        # Node sampling
        self.graph_generator = make_prototype_evaluator(self.cfg)
        # Pre-processing for the vision-to-graph transformation
        self.head_in_cfg = cfg.MODEL.MIDDLE_HEAD.IN_NORM

        if self.head_in_cfg == 'LN':
            self.head_in_ln = build_V2G_linear(cfg)
            init_item.append('head_in_ln')
        else:
            print(self.head_in_cfg)
            import ipdb;
            ipdb.set_trace()
            self.head_in = V2GConv(cfg, in_channels, in_channels, mode='in')

        # node classification layers
        self.node_cls_middle = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        init_item.append('node_cls_middle')

        # Graph-guided Memory Bank
        self.seed_project_left = nn.Linear(256, 256)  # projection layer for the node completion
        self.register_buffer('sr_seed', torch.randn(self.num_classes, 256))  # seed = bank
        self.register_buffer('tg_seed', torch.randn(self.num_classes, 256))

        self.cross_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Cross Graph Interaction

        if self.with_hyper_graph:
            self.intra_domain_graph = HyperGraph(emb_dim=256, K_neigs=self.num_hyper_edge, num_layer=self.num_hypergnn_layer)  # Intra-domain graph aggregation
        else:
            self.intra_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Intra-domain graph aggregation

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=256)
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Structure-aware Matching Loss
        # Different matching loss choices
        if cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'L1':
            self.matching_loss = nn.L1Loss(reduction='sum')
        elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'MSE':
            self.matching_loss = nn.MSELoss(reduction='sum')
        elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'BCE':
            self.matching_loss = BCEFocalLoss()
        self.quadratic_loss = torch.nn.L1Loss(reduction='mean')

        if self.with_node_dis:
            self.grad_reverse = GradientReversal(self.lambda_dis)
            self.node_dis_2 = nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            init_item.append('node_dis')
            self.loss_fn = nn.BCEWithLogitsLoss()
        self._init_weight(init_item)

    def _init_weight(self, init_item=None):
        nn.init.normal_(self.seed_project_left.weight, std=0.01)
        nn.init.constant_(self.seed_project_left.bias, 0)
        if 'node_dis' in init_item:
            for i in self.node_dis_2:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('node_dis initialized')
        if 'node_cls_middle' in init_item:
            for i in self.node_cls_middle:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('node_cls_middle initialized')
        if 'head_in_ln' in init_item:
            for i in self.head_in_ln:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('head_in_ln initialized')

    def forward(self, images, features, targets=None, score_maps=None):
        '''
        We have equal number of source/target feature maps
        features: [sr_feats, tg_feats]
        targets: [sr_targets, None]

        '''
        if targets:
            features, feat_loss = self._forward_train(images, features, targets, score_maps)
            return features, feat_loss

        else:
            features = self._forward_inference(images, features)
            return features, None

    def _forward_train(self, images, features, targets=None, score_maps=None):
        features_s, features_t = features
        middle_head_loss = {}
        # STEP1: sample pixels and generate semantic incomplete graph nodes
        # node_1 and node_2 mean the source/target raw nodes
        # label_1 and label_2 mean the GT and pseudo labels
        nodes_1, labels_1, weights_1 = self.graph_generator(
            self.compute_locations(features_s), features_s, targets
        )
        nodes_2, labels_2, weights_2 = self.graph_generator(
            None, features_t, score_maps
        )

        if nodes_1.size(0) < 6 or len(nodes_1.size()) == 1:
            return features, middle_head_loss

        #  conduct node alignment to prevent overfit
        if self.with_node_dis and nodes_2 is not None and self.node_dis_place == 'feat':
            nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
            target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
            target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
            tg_rev = torch.cat([target_1, target_2], dim=0)
            nodes_rev = self.node_dis_2(nodes_rev)
            node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
            middle_head_loss.update({'dis_loss': node_dis_loss})

        # STEP2: vision-to-graph transformation
        # LN is conducted on the node embedding
        # GN/BN are conducted on the whole image feature
        if self.head_in_cfg != 'LN':
            features_s = self.head_in(features_s)
            features_t = self.head_in(features_t)
            nodes_1, labels_1, weights_1 = self.graph_generator(
                self.compute_locations(features_s), features_s, targets
            )
            nodes_2, labels_2, weights_2 = self.graph_generator(
                None, features_t, score_maps
            )
        else:
            nodes_1 = self.head_in_ln(nodes_1)
            nodes_2 = self.head_in_ln(nodes_2) if nodes_2 is not None else None

        # TODO: Matching can only work for adaptation when both source and target nodes exist.
        # Otherwise, we split the source nodes half-to-half to train SIGMA

        if nodes_2 is not None:  # Both domains have graph nodes

            # STEP3: Conduct Domain-guided Node Completion (DNC)
            (nodes_1, nodes_2), (labels_1, labels_2), (weights_1, weights_2) = \
                self._forward_preprocessing_source_target((nodes_1, nodes_2), (labels_1, labels_2),
                                                          (weights_1, weights_2))

            # STEP4: Single-layer HGCN
            if self.with_complete_graph:
                nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
                nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

            # STEP5: Update Graph-guided Memory Bank (GMB) with enhanced node embedding
            self.update_seed(nodes_1, labels_1, nodes_2, labels_2)

            if self.with_node_dis and self.node_dis_place == 'intra':
                nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                tg_rev = torch.cat([target_1, target_2], dim=0)
                nodes_rev = self.node_dis_2(nodes_rev)
                node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                middle_head_loss.update({'dis_loss': node_dis_loss})

            # STEP6: Conduct Cross Graph Interaction (CGI)
            if self.with_domain_interaction:
                nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)

            if self.with_node_dis and self.node_dis_place == 'inter':
                nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                tg_rev = torch.cat([target_1, target_2], dim=0)
                nodes_rev = self.node_dis_2(nodes_rev)
                node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                middle_head_loss.update({'dis_loss': node_dis_loss})

            # STEP7: Generate node loss
            node_loss = self._forward_node_loss(
                torch.cat([nodes_1, nodes_2], dim=0),
                torch.cat([labels_1, labels_2], dim=0),
                torch.cat([weights_1, weights_2], dim=0)
            )

        else:  # Use all source nodes for training if no target nodes in the early training stage
            (nodes_1, nodes_2), (labels_1, labels_2) = \
                self._forward_preprocessing_source(nodes_1, labels_1)

            nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
            nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

            self.update_seed(nodes_1, labels_1, nodes_1, labels_1)

            nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)
            node_loss = self._forward_node_loss(
                torch.cat([nodes_1, nodes_2], dim=0),
                torch.cat([labels_1, labels_2], dim=0)
            )

        middle_head_loss.update({'node_loss': self.weight_nodes * node_loss})

        # STEP8: Generate Semantic-aware Node Affinity and Structure-aware Matching loss
        if self.matching_cfg != 'none':
            matching_loss_affinity, affinity = self._forward_aff(nodes_1, nodes_2, labels_1, labels_2)
            middle_head_loss.update({'mat_loss_aff': self.weight_matching * matching_loss_affinity})

            if self.with_quadratic_matching:
                matching_loss_quadratic = self._forward_qu(nodes_1, nodes_2, edges_1.detach(), edges_2.detach(), affinity)
                middle_head_loss.update({'mat_loss_qu': matching_loss_quadratic})

        return features, middle_head_loss

    def _forward_preprocessing_source_target(self, nodes, labels, weights):

        '''
        nodes: sampled raw source/target nodes
        labels: the ground-truth/pseudo-label of sampled source/target nodes
        weights: the confidence of sampled source/target nodes ([0.0,1.0] scores for target nodes and 1.0 for source nodes )

        We permute graph nodes according to the class from 1 to K and complete the missing class.

        '''
        sr_nodes, tg_nodes = nodes
        sr_nodes_label, tg_nodes_label = labels
        sr_loss_weight, tg_loss_weight = weights

        labels_exist = torch.cat([sr_nodes_label, tg_nodes_label]).unique()

        sr_nodes_category_first = []
        tg_nodes_category_first = []

        sr_labels_category_first = []
        tg_labels_category_first = []

        sr_weight_category_first = []
        tg_weight_category_first = []

        for c in labels_exist:

            sr_indx = sr_nodes_label == c
            tg_indx = tg_nodes_label == c

            sr_nodes_c = sr_nodes[sr_indx]
            tg_nodes_c = tg_nodes[tg_indx]

            sr_weight_c = sr_loss_weight[sr_indx]
            tg_weight_c = tg_loss_weight[tg_indx]

            if sr_indx.any() and tg_indx.any():  # If the category appear in both domains, we directly collect them!

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                labels_sr = sr_nodes_c.new_ones(len(sr_nodes_c)) * c
                labels_tg = tg_nodes_c.new_ones(len(tg_nodes_c)) * c

                sr_labels_category_first.append(labels_sr)
                tg_labels_category_first.append(labels_tg)

                sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(tg_weight_c)

            elif tg_indx.any():  # If there're no source nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(tg_nodes_c)
                sr_nodes_c = self.sr_seed[c].unsqueeze(0).expand(num_nodes, 256)

                if self.with_semantic_completion:
                    sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).cuda() + sr_nodes_c if len(
                        tg_nodes_c) < 5 \
                        else torch.normal(mean=sr_nodes_c,
                                          std=tg_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())).cuda()
                else:
                    sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).cuda()
                sr_nodes_c = self.seed_project_left(sr_nodes_c)
                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)
                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)
                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)
                sr_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).cuda())
                tg_weight_category_first.append(tg_weight_c)

            elif sr_indx.any():  # If there're no target nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(sr_nodes_c)

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_c = self.tg_seed[c].unsqueeze(0).expand(num_nodes, 256)

                if self.with_semantic_completion:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).cuda() + tg_nodes_c if len(
                        sr_nodes_c) < 5 \
                        else torch.normal(mean=tg_nodes_c,
                                          std=sr_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())).cuda()
                else:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).cuda()

                tg_nodes_c = self.seed_project_left(tg_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)
                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)

                sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).cuda())

        nodes_sr = torch.cat(sr_nodes_category_first, dim=0)
        nodes_tg = torch.cat(tg_nodes_category_first, dim=0)

        weight_sr = torch.cat(sr_weight_category_first, dim=0)
        weight_tg = torch.cat(tg_weight_category_first, dim=0)

        label_sr = torch.cat(sr_labels_category_first, dim=0)
        label_tg = torch.cat(tg_labels_category_first, dim=0)

        return (nodes_sr, nodes_tg), (label_sr, label_tg), (weight_sr, weight_tg)

    def _forward_preprocessing_source(self, sr_nodes, sr_nodes_label):
        labels_exist = sr_nodes_label.unique()

        nodes_1_cls_first = []
        nodes_2_cls_first = []
        labels_1_cls_first = []
        labels_2_cls_first = []

        for c in labels_exist:
            sr_nodes_c = sr_nodes[sr_nodes_label == c]
            nodes_1_cls_first.append(torch.cat([sr_nodes_c[::2, :]]))
            nodes_2_cls_first.append(torch.cat([sr_nodes_c[1::2, :]]))

            labels_side1 = sr_nodes_c.new_ones(len(nodes_1_cls_first[-1])) * c
            labels_side2 = sr_nodes_c.new_ones(len(nodes_2_cls_first[-1])) * c

            labels_1_cls_first.append(labels_side1)
            labels_2_cls_first.append(labels_side2)

        nodes_1 = torch.cat(nodes_1_cls_first, dim=0)
        nodes_2 = torch.cat(nodes_2_cls_first, dim=0)

        labels_1 = torch.cat(labels_1_cls_first, dim=0)
        labels_2 = torch.cat(labels_2_cls_first, dim=0)

        return (nodes_1, nodes_2), (labels_1, labels_2)

    def _forward_intra_domain_graph(self, nodes):
        nodes, edges = self.intra_domain_graph([nodes, nodes, nodes])
        return nodes, edges

    def _forward_cross_domain_graph(self, nodes_1, nodes_2):

        if self.with_global_graph:
            n_1 = len(nodes_1)
            n_2 = len(nodes_2)
            global_nodes = torch.cat([nodes_1, nodes_2], dim=0)
            global_nodes = self.cross_domain_graph(global_nodes, global_nodes, global_nodes)[0]

            nodes1_enahnced = global_nodes[:n_1]
            nodes2_enahnced = global_nodes[n_1:]
        else:
            nodes2_enahnced = self.cross_domain_graph([nodes_1, nodes_1, nodes_2])[0]
            nodes1_enahnced = self.cross_domain_graph([nodes_2, nodes_2, nodes_1])[0]

        return nodes1_enahnced, nodes2_enahnced

    def _forward_node_loss(self, nodes, labels, weights=None):

        labels = labels.long()
        assert len(nodes) == len(labels)

        if weights is None:  # Source domain
            if self.with_cond_cls:
                tg_embeds = self.node_cls_middle(self.tg_seed)
                logits = self.dynamic_fc(nodes, tg_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels,
                                        reduction='mean')
        else:  # Target domain
            if self.with_cond_cls:
                sr_embeds = self.node_cls_middle(self.sr_seed)
                logits = self.dynamic_fc(nodes, sr_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels.long(),
                                        reduction='none')
            node_loss = (node_loss * weights).float().mean() if self.with_score_weight else node_loss.float().mean()

        return node_loss

    def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):

        k = 20  # conduct clustering when we have enough graph nodes
        for cls in sr_labels.unique().long():
            bs = sr_nodes[sr_labels == cls].detach()

            if len(bs) > k and self.with_cluster_update:
                # TODO Use Pytorch-based GPU version
                sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                seed_cls = self.sr_seed[cls]
                indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                indx = (indx == indx[0])[1:]
                bs = bs[indx].mean(0)
            else:
                bs = bs.mean(0)

            momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.sr_seed[cls].unsqueeze(0))
            self.sr_seed[cls] = self.sr_seed[cls] * momentum + bs * (1.0 - momentum)

        if tg_nodes is not None:
            for cls in tg_labels.unique().long():
                bs = tg_nodes[tg_labels == cls].detach()
                if len(bs) > k and self.with_cluster_update:
                    seed_cls = self.tg_seed[cls]
                    sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                    assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                    indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                    indx = (indx == indx[0])[1:]
                    bs = bs[indx].mean(0)
                else:
                    bs = bs.mean(0)
                momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.tg_seed[cls].unsqueeze(0))
                self.tg_seed[cls] = self.tg_seed[cls] * momentum + bs * (1.0 - momentum)

    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2):
        if self.matching_cfg == 'o2o':
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())

            M = self.InstNorm_layer(M[None, None, :, :])
            M = self.sinkhorn_iter(M[:, 0, :, :], n_iters=20).squeeze().exp()

            TP_mask = (matching_target == 1).float()
            indx = (M * TP_mask).max(-1)[1]
            TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

            FP_samples = M[matching_target == 0].view(-1, 1)
            FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()

            # TODO Find a better reduction strategy
            TP_loss = self.matching_loss(TP_samples, TP_target.float()) / len(TP_samples)
            FP_loss = self.matching_loss(FP_samples, FP_target.float()) / torch.sum(FP_samples).detach()
            matching_loss = TP_loss + FP_loss

        elif self.matching_cfg == 'm2m':  # Refer to the Appendix
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())
            matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()
        else:
            M = None
            matching_loss = 0
        return matching_loss, M

    def _forward_inference(self, images, features):
        return features

    def _forward_qu(self, nodes_1, nodes_2, edges_1, edges_2, affinity):

        if self.with_hyper_graph:

            # hypergraph matching (high order)
            translated_indx = list(range(1, self.num_hyper_edge))+[int(0)]
            mathched_index = affinity.argmax(0)
            matched_node_1 = nodes_1[mathched_index]
            matched_edge_1 = edges_1.t()[mathched_index]
            matched_edge_1[matched_edge_1 > 0] = 1

            matched_node_2 =nodes_2
            matched_edge_2 =edges_2.t()
            matched_edge_2[matched_edge_2 > 0] = 1
            n_nodes = matched_node_1.size(0)

            angle_dis_list = []
            for i in range(n_nodes):
                triangle_1 = nodes_1[matched_edge_1[i, :].bool()]  # 3 x 256
                triangle_1_tmp = triangle_1[translated_indx]
                # print(triangle_1.size(), triangle_1_tmp.size())
                sin1 = torch.sqrt(1.- F.cosine_similarity(triangle_1, triangle_1_tmp).pow(2)).sort()[0]
                triangle_2 = nodes_2[matched_edge_2[i, :].bool()]  # 3 x 256
                triangle_2_tmp = triangle_2[translated_indx]
                sin2 = torch.sqrt(1.- F.cosine_similarity(triangle_2, triangle_2_tmp).pow(2)).sort()[0]
                angle_dis = (-1 / self.angle_eps  * (sin1 - sin2).abs().sum()).exp()
                angle_dis_list.append(angle_dis.view(1,-1))

            angle_dis_list = torch.cat(angle_dis_list)
            loss = angle_dis_list.mean()
        else:
            # common graph matching (2nd order)
            R = torch.mm(edges_1, affinity) - torch.mm(affinity, edges_2)
            loss = self.quadratic_loss(R, R.new_zeros(R.size()))
        return loss

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

    def sinkhorn_iter(self, log_alpha, n_iters=5, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()
        return log_alpha

    def dynamic_fc(self, features, kernel_par):
        weight = kernel_par
        return torch.nn.functional.linear(features, weight, bias=None)

    def dynamic_conv(self, features, kernel_par):
        weight = kernel_par.view(self.num_classes, -1, 1, 1)
        return torch.nn.functional.conv2d(features, weight)

    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long(), :].cuda()


def build_graph_matching_head(cfg, in_channels):
    return GModule(cfg, in_channels)