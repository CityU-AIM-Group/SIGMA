import torch
import torch.nn as nn
import numpy as np

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

    def forward(self, key, value, query, attn_mask=None):

        if self.version == 'v2':
            B =1
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
            context = context.transpose(0, 1).contiguous().view(query.size(1), B, dim_per_head * num_heads)
            output = self.linear_final(context)
            # dropout
            output = self.dropout(output)
            output = self.layer_norm(residual + output)
            # output = residual + output

        elif self.version == 'v1': # some difference about the place of torch.view fuction
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
            context = context.view(batch_size, -1, dim_per_head * num_heads)
            output = self.linear_final(context)
            output = self.dropout(output)
            output = self.layer_norm(residual + output)

        return output.squeeze(), attention.squeeze()

class CrossGraph(nn.Module):
    """ This class hasn't been used"""
    def __init__(self, model_dim=256,  dropout=0.0,):
        super(CrossGraph, self).__init__()


        self.linear_node1 = nn.Linear(model_dim,model_dim)
        self.linear_node2 = nn.Linear(model_dim,model_dim)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)         


    def forward(self, node_1, node_2,  attn_mask=None):
        node_1_r = node_1
        node_2_r = node_2

        edge1 = self.linear_edge(node_1)
        edge2 = self.linear_edge(node_2)

        node_1_ = self.linear_node1(node_1)
        node_2_ = self.linear_node1(node_2)

        attention = torch.mm(edge1,edge2.t())

        node_1 = torch.mm(attention.softmax(-1), node_2_)
        node_2 = torch.mm(attention.t().softmax(-1), node_1_)


        node_1 = self.linear_final(node_1)
        node_2 = self.linear_final(node_2)

        node_1 = self.dropout(node_1)
        node_2  = self.dropout(node_2)
        node_1 = self.layer_norm(node_1_r + node_1)

        node_2 = self.layer_norm(node_2_r + node_2)


        return node_1, node_2