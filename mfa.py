import torch
import torch.nn as nn


class MFA(nn.Module):
    def __init__(self, node_nums=90):
        super(MFA, self).__init__()
        self.node_nums = node_nums
        self.relu = nn.ReLU(inplace=True)
        self.a = nn.Parameter(torch.empty((self.node_nums, 1))).cuda()
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x_co, x_sc, x_fc):
        x_co_sc = torch.mul(x_co, x_sc)
        x_co_fc = torch.mul(x_co, x_fc)
        x_co_sf = torch.mul(self.a, x_co_sc) + torch.mul(1 - self.a, x_co_fc)
        out = torch.cat([x_co, x_co_sf], dim=1)
        return out


# 各模态的特异性特征与共性特征直接拼接
class MFA_ab1(nn.Module):
    def __init__(self, node_nums=90):
        super(MFA_ab1, self).__init__()
        self.node_nums = node_nums
        self.relu = nn.ReLU(inplace=True)
        self.a = nn.Parameter(torch.empty((self.node_nums, 1))).cuda()
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x_co, x_sc, x_fc):
        x_co_sc = torch.mul(x_co, x_sc)
        x_co_fc = torch.mul(x_co, x_fc)
        # x_co_sf = torch.mul(self.a, x_co_sc) + torch.mul(1 - self.a, x_co_fc)
        out = torch.cat([x_co, x_co_sc, x_co_fc], dim=1)
        return out


class MFA_ab2(nn.Module):
    def __init__(self, node_nums=90):
        super(MFA_ab2, self).__init__()
        self.node_nums = node_nums
        self.relu = nn.ReLU(inplace=True)
        self.a = nn.Parameter(torch.empty((self.node_nums, 1))).cuda()
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x_co, x_sc, x_fc):
        out = torch.cat([x_co, x_sc, x_fc], dim=1)
        return out


class MFA1(nn.Module):
    def __init__(self, node_nums=90):
        super(MFA1, self).__init__()
        self.node_nums = node_nums
        self.relu = nn.ReLU(inplace=True)
        self.a = nn.Parameter(torch.empty((self.node_nums, 1))).cuda()
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x_sc, x_fc):
        out = torch.cat([x_sc, x_fc], dim=1)
        return out


class MFA1_ab1(nn.Module):
    def __init__(self, node_nums=90):
        super(MFA1_ab1, self).__init__()
        self.node_nums = node_nums
        self.relu = nn.ReLU(inplace=True)
        self.a = nn.Parameter(torch.empty((self.node_nums, 1))).cuda()
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x_sc, x_fc):
        max_features = torch.max(x_sc, x_fc)
        multi_features = torch.mul(x_sc, x_fc)
        out = torch.cat([max_features, multi_features], dim=1)
        return out


class MFA2(nn.Module):
    def __init__(self, node_nums=90):
        super(MFA2, self).__init__()
        self.node_nums = node_nums
        self.relu = nn.ReLU(inplace=True)
        self.a = nn.Parameter(torch.empty((self.node_nums, 1))).cuda()
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x_sc, x_fc):
        multi_feats = torch.cat([x_sc, x_fc], dim=-1)
        mean_feats = torch.mean(multi_feats, dim=1, keepdim=True)
        max_feats = torch.max(multi_feats, dim=1, keepdim=True)[0]
        sum_feats = torch.sum(multi_feats, dim=1, keepdim=True)
        out = torch.cat([mean_feats, max_feats, sum_feats], dim=1)
        return out
