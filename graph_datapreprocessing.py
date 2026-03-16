import numpy as np
import torch
from torch_geometric.data import Data
import os
import scipy.io as sio
import pickle
import scipy.io as sio
# path = r'/home/hy/BrainGraph/Dataset/structure_connect/'
fpath = r'/home/hy/BrainGraph/Dataset/function_connect/'
apath = r'/home/hy/BrainGraph/Dataset/function/'
# alist = os.listdir(apath)
spath = r'/home/hy/BrainGraph/Dataset/'
flist = os.listdir(fpath)
flist.sort()
# slist = os.listdir(spath)
all_graph = []
for i in range(len(flist)):
    fname = flist[i]
    # sname = fname.split('-')[1][:-4]+'.txt'
    fsub_path = fpath + fname
    edges = np.loadtxt(fsub_path)
    # 创建特征
    sname = fname[:-4]
    # aname = alist[i]
    asub_apth = apath + 'ROICorrelation_sub_' + sname + '.mat'
    atts = sio.loadmat(asub_apth)
    sub_features = atts['cut']
    sub_features = np.array(sub_features)
    # sub_features = np.eye(90)
    features = [feature for feature in sub_features]
    tensor_features = torch.tensor(features, dtype=torch.float)

    source = edges[:, 0]
    target = edges[:, 1]
    edge_index = [source, target]
    tensor_edge_index = torch.tensor(edge_index, dtype=torch.float)
    node_labels = np.zeros(90)
    tensor_node_labels = torch.tensor(node_labels, dtype=torch.long)
    graph = Data(x=tensor_features, edge_index=tensor_edge_index)
    all_graph.append(graph)

    print('finish: {} graph'.format(i+1))

with open(spath + "function2.pkl", "wb") as f:
    pickle.dump(all_graph, f)

print('done!!!')

# attr_path = r'F:/Data/AD_wj/Attrs/'
# dti_path = r'F:/Data/AD_wj/DTI/'
# save_path = r'E:/GAE/data/'
#
# """----------------AD数据 转 Graph-----------------------"""
# attributes = np.load(dti_path + 'DTI148_AD_MCI.npy')
# structure = np.load(dti_path + 'DTI148_AD_MCI.npy')
# all_graph = []
# for i in range(attributes.shape[0]):
#     features = attributes[i]
#     matrix = structure[i]
#     # 将矩阵中非零元素的坐标转换成源节点目标节点序列
#     edge_index = np.nonzero(matrix)  # 以元组的形式返回矩阵中元素的索引，第一个元素是行索引，第二个元素是列索引
#     edge_index = list(edge_index)
#     tensor_feature = torch.tensor(features, dtype=torch.float)
#     tensor_edge_index = torch.tensor(edge_index, dtype=torch.long)
#     graph = Data(x=tensor_feature, edge_index=tensor_edge_index)  # 用于构造图数据，它包含两个属性：x 和 edge_index
#     all_graph.append(graph)
#
# with open(save_path+"AD_MCI_graph_DTI.pkl", "wb") as f:  # wb 表示以二进制格式写入文件
#     pickle.dump(all_graph, f)  # 将 all_graph 对象序列化成字符串
# print('done!!!')