import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TopKPooling, SAGPooling
# from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import softmax, scatter, cumsum
from torch_geometric.nn.inits import uniform


class PoolScore(nn.Module):
    def __init__(self, dim):
        super(PoolScore, self).__init__()
        self.dim = dim  # 由于拼接上了cross-enhanced representations, 这里的dim实则为hidden_dim的2倍

        # self.select = SelectTopK(dim, ratio)
        self.act = activation_resolver("tanh")  # 根据用户的需求或模型的特定要求，选择合适的激活函数（如 ReLU、Sigmoid、Tanh 等）
        self.weight = torch.nn.Parameter(torch.empty(dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.dim, self.weight)

    def forward(self, x):
        x = x.view(-1, 1) if x.dim() == 1 else x
        # score = (x * self.weight).sum(dim=-1)
        score = torch.matmul(x, self.weight)
        score = torch.squeeze(score, -1)
        score = self.act(score / self.weight.norm(p=2, dim=0))
        return score


class MultiG_pooling(nn.Module):
    def __init__(self, num_nodes, num_features, ratio):
        super(MultiG_pooling, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.ratio = ratio
        self.pooling = PoolScore(4 * num_features)
        self.multi_weight = nn.Parameter(torch.empty(2, 1))
        self.multi_bias = nn.Parameter(torch.zeros(num_nodes))
        self.act = activation_resolver("sigmoid")

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform(2, self.multi_weight)
        uniform(self.num_nodes, self.multi_bias)

    def forward(self, x_sc, x_fc):
        sc_score = self.pooling(x_sc).view(-1,90,1)
        fc_score = self.pooling(x_fc).view(-1,90,1)
        all_score = torch.cat((sc_score, fc_score), dim=2)
        multi_score = torch.matmul(all_score, self.multi_weight)
        multi_score = torch.squeeze(multi_score, dim=-1)
        multi_score = self.act(multi_score + self.multi_bias)
        node_index = topk(multi_score, self.ratio, self.num_nodes)
        ex_multi_score = multi_score.unsqueeze(-1).expand(-1, -1, x_sc.size(-1))
        mx_sc = x_sc.view(-1, self.num_nodes, 4*self.num_features)
        mx_fc = x_fc.view(-1, self.num_nodes, 4*self.num_features)
        wx_sc = mx_sc * ex_multi_score
        wx_fc = mx_fc * ex_multi_score
        select_wx_sc = torch.gather(wx_sc, 1, node_index.unsqueeze(-1).expand(-1, -1, x_sc.size(-1)))
        select_wx_fc = torch.gather(wx_fc, 1, node_index.unsqueeze(-1).expand(-1, -1, x_fc.size(-1)))

        return select_wx_sc, select_wx_fc


def topk(score, ratio, num_nodes):
    k = int(np.ceil(ratio * num_nodes))  # 返回 4，因为它向上取整到最接近的整数
    score1, score_idx = torch.sort(score, dim=1, descending=True)
    keep_idx = score_idx[:, :k]
    return keep_idx


if __name__ == "__main__":
    num_nodes = 20
    num_features = 8
    num_classes = 7
    X = torch.randn((8, num_nodes, 2*num_features))
    edge_index = torch.randint(0, num_nodes, (2, 2 * num_nodes))
    topk_pool = MultiG_pooling(X.size(1), num_features, ratio=0.5)
    selct_x_sc, selct_x_fc, select_score = topk_pool(X, X)

    # x = torch.tensor(torch.randn((3, 4)), dtype=torch.float)  # 节点特征
    # edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # 边的索引
    # data = Data(x=x, edge_index=edge_index)
    # k = 2  # 选择前 k 个节点
    # pooling = TopKPooling(in_channels=1, ratio=0.5)  # ratio 表示保留的节点比例
    # # 进行池化操作
    # out, edge_index, _, batch, perm = pooling(data.x, data.edge_index)

    print("done")
