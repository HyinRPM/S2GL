import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils import init_symmetric_weights, init_diagonal_weights


class DimAlign(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DimAlign, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.in_dim != self.out_dim:
            x = self.conv(x)
        else:
            x = x
        return x


class CAI(nn.Module):
    def __init__(self, feat_dim=32, num_node=90):
        super(CAI, self).__init__()
        self.linear_e = nn.Linear(feat_dim, feat_dim, bias=False)
        # self.weight = init_symmetric_weights(feat_dim)
        # self.weight = init_diagonal_weights(feat_dim)
        self.feat_dim = feat_dim
        self.num_node = num_node
        self.align_model = DimAlign(feat_dim, feat_dim)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0, 0.01)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def co_attention(self, sc_features, fc_features):
        sc_features = sc_features.cuda()
        fc_features = fc_features.cuda()
        # self.weight = self.weight.to(self.device)

        exemplar = sc_features.view(-1, self.num_node, self.feat_dim)  # reshape to [batch_size, num_nodes, out_features]
        query = fc_features.view(-1, self.num_node, self.feat_dim)
        # 维度重排
        exemplar = exemplar.permute(0, 2, 1)  # [-1,2,90]
        query = query.permute(0, 2, 1)  # [-1,2,90]

        exemplar_t = torch.transpose(exemplar, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        # exemplar_corr = torch.matmul(exemplar_t, self.weight)  # matmul利用python 中的广播机制
        A = torch.bmm(exemplar_corr, query)  # bmm 要求维度必须相同
        A1 = F.softmax(A.clone(), dim=1)  # 是对 1 这个维度softmax归一化
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar, A1).contiguous()  # 注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query, B).contiguous()
        cosc_features = torch.transpose(query_att, 1, 2).contiguous()
        cofc_features = torch.transpose(exemplar_att, 1, 2).contiguous()
        cosc_features = cosc_features.view(-1, self.feat_dim)  # reshape to [batch_size, num_nodes, out_features]
        cofc_features = cofc_features.view(-1, self.feat_dim)
        return cosc_features, cofc_features

    def forward(self, sc_feats, fc_feats):
        cosc_features, cofc_features = self.co_attention(sc_feats, fc_feats)
        return cosc_features, cofc_features
