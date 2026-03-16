# weighted adjacency matrix as node features

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
#SAG Pooling
class SAGPoolScore(nn.Module):
    def __init__(self, dim, **kwargs):
        super(SAGPoolScore, self).__init__()
        self.dim = dim  # 由于拼接上了cross-enhanced representations, 这里的dim实则为hidden_dim的2倍
        self.gnn = GraphConv(dim, 1, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index):
        x = x.view(-1, 1) if x.dim() == 1 else x
        # score = (x * self.weight).sum(dim=-1)
        attn = self.gnn(x, edge_index)
        return attn