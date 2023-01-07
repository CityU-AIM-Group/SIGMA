import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

class dot_attention(nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.version  = version
    def forward(self, key_value_query, attn_mask=None):
        if self.version == 'v2':
            B =1
            key, value, query = key_value_query
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            query = query.unsqueeze(1)
            residual = query

            dim_per_head = self.dim_per_head
            num_heads = self.num_heads

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(key.size(0), B * num_heads, dim_per_head).transpose(0,1)
            value = value.view(value.size(0), B * num_heads, dim_per_head).transpose(0,1)
            query = query.view(query.size(0), B * num_heads, dim_per_head).transpose(0,1)

            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            # (query, key, value, scale, attn_mask)
            context = context.transpose(0, 1).contiguous().view(query.size(1), B, dim_per_head * num_heads) # set2
            output = self.linear_final(context)
            output = self.dropout(output)
            output = self.layer_norm(residual + output)
        elif self.version == 'v1':
            key, value, query = key_value_query

            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            query = query.unsqueeze(0)
            residual = query
            B, L, C = key.size()
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)

            if attn_mask:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)
            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            context = context.view(batch_size, -1, dim_per_head * num_heads) # set1: directly use 'view'
            output = self.linear_final(context)
            output = self.dropout(output)
            output = self.layer_norm(residual + output)

        return output.squeeze(), attention.squeeze()

class HyperGraph(nn.Module):
    '''
    Feng et al. 'Hypergraph Neural Networks'
    https://arxiv.org/pdf/1809.09401.pdf
    https://github.com/iMoonLab/HGNN
    '''
    def __init__(self, emb_dim=256, K_neigs=[3], num_layer=1, distance_metrix='Eu_dis'):
        super(HyperGraph, self).__init__()
        self.K_neigs = K_neigs if isinstance(K_neigs,list) else [K_neigs]
        self.distance_metric = distance_metrix
        if num_layer == 1:
            self.hgnn_conv_layer = single_layer_HGNN_conv(emb_dim, emb_dim, bias=True)
        else:
            self.hgnn_conv_layer = double_layer_HGNN_conv(emb_dim, emb_dim//2, bias=True)

    def forward(self, node_feat):
        # Establish hypergraph with KNN
        node_feat = node_feat[0]
        with torch.no_grad():
            tmp = self.construct_H_with_KNN(node_feat.detach().cpu().numpy())
            H = self.hyperedge_concat(None, tmp)
            G = self.generate_G_from_H(H)
            G = torch.Tensor(G).to(node_feat.device)
            H = torch.Tensor(H).to(node_feat.device)
        outputs = self.hgnn_conv_layer(node_feat, G)
        return outputs, H

    def generate_G_from_H(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        if type(H) != list:
            return self._generate_G_from_H(H, variable_weight)
        else:
            G = []
            for sub_H in H:
                G.append(self.generate_G_from_H(sub_H, variable_weight))
            return G

    def hyperedge_concat(self, *H_list):
        """
        Concatenate hyperedge group in H_list
        :param H_list: Hyperedge groups which contain two or more hyper-graph incidence matrix
        :return: Fused hypergraph incidence matrix
        """
        H = None
        for h in H_list:
            if h is not None and h != []:
                # for the first H appended to fused hypergraph incidence matrix
                if H is None:
                    H = h
                else:
                    if type(h) != list:
                        H = np.hstack((H, h))
                    else:
                        tmp = []
                        for a, b in zip(H, h):
                            tmp.append(np.hstack((a, b)))
                        H = tmp
        return H

    def construct_H_with_KNN(self, X, split_diff_scale=False, is_probH=True, m_prob=1):
        """
        init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
        :param X: N_object x feature_number
        :param K_neigs: the number of neighbor expansion
        :param split_diff_scale: whether split hyperedge group at different neighbor scale
        :param is_probH: prob Vertex-Edge matrix or binary
        :param m_prob: prob
        :return: N_object x N_hyperedge
        """
        if len(X.shape) != 2:
            X = X.reshape(-1, X.shape[-1])
        if type(self.K_neigs) == int:
            K_neigs = [self.K_neigs]
        dis_mat = Eu_dis(X)
        H = []
        for k_neig in self.K_neigs:
            H_tmp = self.construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
            if not split_diff_scale:
                H = self.hyperedge_concat(H, H_tmp)
            else:
                H.append(H_tmp)
        return H

    def construct_H_with_KNN_from_distance(self, dis_mat, k_neig, is_probH=True, m_prob=1):
        """
        construct hypregraph incidence matrix from hypergraph node distance matrix
        :param dis_mat: node distance matrix
        :param k_neig: K nearest neighbor
        :param is_probH: prob Vertex-Edge matrix or binary
        :param m_prob: prob
        :return: N_object X N_hyperedge
        """
        n_obj = dis_mat.shape[0]
        # construct hyperedge from the central feature space of each node
        n_edge = n_obj
        H = np.zeros((n_obj, n_edge))
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx
            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                else:
                    H[node_idx, center_idx] = 1.0
        return H

    def _generate_G_from_H(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            return G

class single_layer_HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, dropout=0.5, bias=True):
        super(single_layer_HGNN_conv, self).__init__()
        self.linear = nn.Linear(in_ft, out_ft, bias=bias)
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        nn.init.normal_(self.linear.weight, std=stdv)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = self.linear(x)
        x = G.matmul(x)
        return x

class double_layer_HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, dropout=0.5, bias=True):
        super(double_layer_HGNN_conv, self).__init__()
        self.linear1 = nn.Linear(in_ft, out_ft, bias=bias)
        self.linear2 = nn.Linear(out_ft, in_ft, bias=bias)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear1.weight.size(1))
        nn.init.normal_(self.linear1.weight, std=stdv)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.weight, std=stdv)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = self.linear1(x)
        x = F.relu(G.matmul(x))
        x = F.dropout(x, self.dropout)
        x = self.linear2(x)
        x = F.relu(G.matmul(x))
        return x