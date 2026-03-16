import random

import torch
import torch.nn as nn
from mgae import GCN_mgaev3
from cai import CAI
from mfa import MFA2
import torch.nn.functional as F
from multi_pooling import MultiG_pooling
import numpy as np

class FCNN_Classifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 dropout):
        super(FCNN_Classifier, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            hidden_dim1 = hidden_dim // 2
            self.layers.append(
                nn.Linear(hidden_dim, hidden_dim1))
            # self.layers.append(nn.Sequential(
            #     nn.Linear(hidden_dim, hidden_dim1),
            #     nn.BatchNorm1d(hidden_dim1, track_running_stats=False),
            #     nn.Tanh()))
            hidden_dim = hidden_dim1
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        x = F.tanh(x)
        return x


class CA_MGAE2(nn.Module):
    def __init__(self, args):  # sfl_block refers to mgae
        super(CA_MGAE2, self).__init__()
        self.args = args
        self.encoder_layers = args.encoder_layers
        self.encoder_dim = args.encoder_dim
        self.keep_nodes = int(np.ceil(args.pool_ratio * args.num_nodes))
        self.encoder = GCN_mgaev3(args.num_features, args.encoder_dim, args.encoder_dim, args.encoder_layers,
                                  args.dropout)
        self.mfa_block = MFA2()
        self.multi_pool = MultiG_pooling(num_nodes=args.num_nodes, num_features=args.encoder_dim,
                                         ratio=args.pool_ratio)
        self.clasifier = FCNN_Classifier(8 * 3 * args.encoder_dim, args.fcnn_dim, args.num_classes,
                                         args.num_fcnnlayers, dropout=args.dropout)

    def forward(self, struc_graph, func_graph, pre_struc_edge, pre_func_edge, batch_size):
        struc_feature_list, func_feature_list = self.encoder(struc_graph.x, pre_struc_edge, func_graph.x, pre_func_edge)
        multilayer_struc_feats, multilayer_func_feats = self.multilevel_pool(struc_feature_list, func_feature_list)

        # multilayer_struc_feats = torch.cat(pstruc_feature_list, dim=-1)
        # multilayer_func_feats = torch.cat(pfunc_feature_list, dim=-1)
        mfa_features = self.mfa_block(multilayer_struc_feats, multilayer_func_feats)

        mfa_features = mfa_features.view(batch_size, -1)
        x = self.clasifier(mfa_features)
        return x, struc_feature_list, func_feature_list

    def multilevel_pool(self, struc_feature_list,  func_feature_list):
        all_struc_features = torch.cat(struc_feature_list, dim=-1)
        all_func_features = torch.cat(func_feature_list, dim=-1)
        wt_struc_features, wt_func_features = self.multi_pool(all_struc_features, all_func_features)
        return wt_struc_features, wt_func_features


    # def multilevel_pool(self, struc_feature_list, func_feature_list):
    #     pstruc_feature_list = []
    #     pfunc_feature_list = []
    #
    #     for i in range(self.encoder_layers):
    #         rstruc_feature = struc_feature_list[i].view(-1, 90, 2*self.encoder_dim)
    #         rfunc_feature = func_feature_list[i].view(-1, 90,  2*self.encoder_dim)
    #         struc_feature, func_feature = self.multi_pool(rstruc_feature, rfunc_feature)
    #
    #         pstruc_feature_list.append(struc_feature)
    #         pfunc_feature_list.append(func_feature)
    #     return pstruc_feature_list, pfunc_feature_list
