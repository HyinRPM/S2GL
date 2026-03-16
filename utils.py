import numpy as np
import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from torch_geometric.nn.pool import topk_pool, sag_pool
from torch_geometric.nn import TopKPooling
# 无向掩码
def edgemask_um(mask_ratio, split_edge, device, num_nodes):
    if split_edge.size(0) == 2:
        edge_index = split_edge.t()
    else:
        edge_index = split_edge
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].t()
    edge_index = to_undirected(edge_index_train)  #
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)


# 有向掩码
def edgemask_dm(mask_ratio, split_edge, device, num_nodes):
    if split_edge.size(0) == 2:
        split_edge = split_edge.t()
    else:
        split_edge = split_edge
    if isinstance(split_edge, torch.Tensor):
        edge_index = to_undirected(split_edge.t()).t()
    else:
        edge_index = torch.stack([split_edge['train']['edge'][:, 1], split_edge['train']['edge'][:, 0]], dim=1)
        edge_index = torch.cat([split_edge['train']['edge'], edge_index], dim=0)

    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index = edge_index_train
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)


def mask_edges(data, args, mask_type):
    features = data.x
    edge_index = data.edge_index.type(torch.int64)
    # mask edges
    if mask_type == 'um':
        adj, edge_index, edge_index_mask = edgemask_um(args.mask_ratio, edge_index, features.device,
                                                       data.num_nodes)
    else:
        adj, edge_index, edge_index_mask = edgemask_dm(args.mask_ratio, edge_index, features.device,
                                                       data.num_nodes)
    pre_edge_index = adj.to(features.device)  # no mask edges
    pos_train_edge = edge_index_mask
    return pre_edge_index, pos_train_edge, edge_index


def lp_loss(decoder, h, pos_train_edge):
    edge = pos_train_edge
    pos_out = decoder(h, edge)
    pos_loss = -torch.log(pos_out + 1e-15).mean()
    # pos_loss = torch.nn.CrossEntropyLoss()

    # pos_edge = split_edge['train']['edge'].t()
    # new_edge_index, _ = add_self_loops(edge_index.cpu())
    # edge = negative_sampling(
    #     new_edge_index, num_nodes=args.num_nodes,
    #     num_neg_samples=pos_train_edge.shape[1])
    # # num_neg_samples 指定每个正训练边要生成的负样本数量
    # # edge = edge.to(features.device)
    #
    # neg_out = decoder(h, edge)
    # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
    # loss = pos_loss + neg_loss
    return pos_loss


def lp_loss1(decoder, h, pos_train_edge, edge_index, args):
    edge = pos_train_edge
    pos_out = decoder(h, edge)
    pos_loss = -torch.log(pos_out + 1e-15).mean()
    # pos_loss = torch.nn.CrossEntropyLoss()

    # pos_edge = split_edge['train']['edge'].t()
    new_edge_index, _ = add_self_loops(edge_index.cpu())
    edge = negative_sampling(
        new_edge_index, num_nodes=args.num_nodes,
        num_neg_samples=pos_train_edge.shape[1])
    # num_neg_samples 指定每个正训练边要生成的负样本数量
    # edge = edge.to(features.device)

    neg_out = decoder(h, edge)
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
    loss = pos_loss + neg_loss
    return loss


def init_symmetric_weights(dim):
    W = torch.randn(dim, dim) * 0.01
    symmetric_matrix = (W + W.t()) / 2
    return nn.Parameter(symmetric_matrix.cuda())


def init_diagonal_weights(dim):
    diag_values = torch.randn(dim) * 0.01
    diagonal_matrix = torch.diag(diag_values).cuda()
    weight_matrix = nn.Parameter(diagonal_matrix)
    return weight_matrix


