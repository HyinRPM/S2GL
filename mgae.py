import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
from utils import edgemask_um, edgemask_dm
from cai import CAI


class GCN_mgaev3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_mgaev3, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(2*hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout
        self.co_att = CAI(out_channels, num_node=90)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_sc, adj_sc, x_fc, adj_fc):
        xx_sc = []
        xx_fc = []
        for conv in self.convs[:-1]:
            x_sc = conv(x_sc, adj_sc)
            x_sc = F.relu(x_sc)
            x_sc = F.dropout(x_sc, p=self.dropout, training=self.training)

            x_fc = conv(x_fc, adj_fc)
            x_fc = F.relu(x_fc)
            x_fc = F.dropout(x_fc, p=self.dropout, training=self.training)
            cosc_features, cofs_features = self.co_att(x_sc, x_fc)
            # x_sc = sum([x_sc, cosc_features])
            # x_fc = sum([x_fc, cofs_features])
            x_sc = torch.cat((x_sc, cosc_features), dim=1)
            x_fc = torch.cat((x_fc, cofs_features), dim=1)
            xx_sc.append(x_sc)
            xx_fc.append(x_fc)

        x_sc = self.convs[-1](x_sc, adj_sc)
        x_sc = F.relu(x_sc)
        x_fc = self.convs[-1](x_fc, adj_fc)
        x_fc = F.relu(x_fc)
        cosc_features, cofs_features = self.co_att(x_sc, x_fc)
        # x_sc = sum([x_sc, cosc_features])
        # x_fc = sum([x_fc, cofs_features])
        x_sc = torch.cat((x_sc, cosc_features), dim=1)
        x_fc = torch.cat((x_fc, cofs_features), dim=1)
        xx_sc.append(x_sc)
        xx_fc.append(x_fc)

        return xx_sc, xx_fc

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, de_v='v1'):
        super(LPDecoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        if de_v == 'v1':
            self.lins.append(torch.nn.Linear(in_channels * n_layer*2, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)






