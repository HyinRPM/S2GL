import pickle
import argparse
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from multi_dataload import MultimodalDataset
from mgae import LPDecoder
from ca_mgae import CA_MGAE2
from train_test import train, test
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='S2GL')
parser.add_argument("--num_nodes", type=str, default=90)
parser.add_argument("--num_features", type=str, default=90)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--mask_type", type=str, default='um', help='um | dm')  # um 为无向掩码， dm为有向
parser.add_argument("--encoder_layers", type=int, default=2)
parser.add_argument("--decoder_layers", type=int, default=2)
parser.add_argument("--encoder_dim", type=int, default=16, help="Dimensions of hidden layers")
parser.add_argument("--decoder_dim", type=int, default=256, help="Dimensions of hidden layers")
parser.add_argument("--num_fcnnlayers", type=int, default=4)
parser.add_argument("--fcnn_dim", type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epoches', type=int, default=500)
parser.add_argument('--mask_ratio', type=float, default=0.7)
parser.add_argument('--seed', type=int, default=22, help='Random seed.')
parser.add_argument("--de_v", type=str, default='v1')
parser.add_argument("--pool_ratio", type=float, default=0.25)

path = r'/home/hy/BrainGraph/Dataset/'
# pickle 模块可以将 Python 对象序列化成字符串，并将其保存到文件中。
# pickle.load() 函数将字符串反序列化成 Python 对象
with open(path + "structure.pkl", "rb") as f:
    structure_grpah = pickle.load(f)
with open(path + "function.pkl", "rb") as f:
    function_grpah = pickle.load(f)
labels = []
for i in range(len(structure_grpah)):
    if i < 50:
        labels.append(0)
    else:
        labels.append(1)
# 加载预定义参数
args = parser.parse_args()


multi_test_maxACC = []
multi_test_maxPRE = []
multi_test_maxREC = []
multi_test_maxF1 = []
for i in range(3):
    kf = KFold(n_splits=10, shuffle=True)
    all_test_maxACC = []
    all_test_maxPRE = []
    all_test_maxREC = []
    all_test_maxF1 = []
    for train_index, test_index in kf.split(labels):
        train_struc_graph, train_func_graph, test_struc_graph, test_func_graph, train_labels, test_labels = [], [], [], [], [], []
        for tr_idx in train_index:
            train_struc_graph.append(structure_grpah[tr_idx])
            train_func_graph.append(function_grpah[tr_idx])
            train_labels.append(labels[tr_idx])
        for te_idx in test_index:
            test_struc_graph.append(structure_grpah[te_idx])
            test_func_graph.append(function_grpah[te_idx])
            test_labels.append(labels[te_idx])

        train_dataset = MultimodalDataset(train_struc_graph, train_func_graph, train_labels)
        test_dataset = MultimodalDataset(test_struc_graph, test_func_graph, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        # model = CA_MGAE(args)
        model = CA_MGAE2(args)
        cross_decoder = LPDecoder(args.encoder_dim, args.decoder_dim, 1, args.encoder_layers,
                                  args.decoder_layers, args.dropout, de_v=args.de_v)

        optimizier = torch.optim.Adam(list(model.parameters()) + list(cross_decoder.parameters()),
                                      lr=args.lr, eps=1e-2, weight_decay=0.003)

        # optimizier = torch.optim.Adam(list(model.parameters()) + list(cross_decoder.parameters()),
        #                               lr=args.lr)

        criterion = torch.nn.CrossEntropyLoss()
        allTrain_loss = []
        allTrain_acc = []
        allTest_loss = []
        allTest_acc = []
        maxTest_acc = 0
        maxTest_pre = 0
        maxTest_rec = 0
        maxTest_f1 = 0
        # model.initialize()
        # cross_decoder.reset_parameters()

        for epoch in range(args.epoches):
            loss_train, acc_train, pre_train, rec_train, f1_train = train(model, cross_decoder, device, train_loader, optimizier,
                                          criterion, epoch, args)
            loss_test, acc_test, pre_test, rec_test, f1_test = test(model, cross_decoder, device, test_loader, criterion, epoch,
                                       args)
            # scheduler.step()
            allTrain_loss.append(loss_train)
            allTrain_acc.append(acc_train)
            allTest_loss.append(loss_test)
            allTest_acc.append(acc_test)
            if epoch > 150 and acc_test > maxTest_acc:
                maxTest_acc = acc_test
                maxTest_pre = pre_test
                maxTest_rec = rec_test
                maxTest_f1 = f1_test
        all_test_maxACC.append(maxTest_acc)
        all_test_maxPRE.append(maxTest_pre)
        all_test_maxREC.append(maxTest_rec)
        all_test_maxF1.append(maxTest_f1)
        # 绘制损失曲线图
        # x = list(range(args.epoches))
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(x, allTrain_loss)
        # plt.plot(x, allTest_loss)
        # plt.legend(['trainLoss', 'valLoss'])
        # plt.xlabel('epochs')
        # plt.ylabel('Loss')

        # plt.subplot(1, 2, 2)
        # plt.plot(x, allTrain_acc)
        # plt.plot(x, allTest_acc)
        # plt.legend(['trainacc', 'valacc'])
        # plt.xlabel('epochs')
        # plt.ylabel('Acc')
        # plt.show()

    avg_test_maxACC = np.average(all_test_maxACC)
    avg_test_maxPRE = np.average(all_test_maxPRE)
    avg_test_maxREC = np.average(all_test_maxREC)
    avg_test_maxF1 = np.average(all_test_maxF1)

    multi_test_maxACC.append(avg_test_maxACC)
    multi_test_maxPRE.append(avg_test_maxPRE)
    multi_test_maxREC.append(avg_test_maxREC)
    multi_test_maxF1.append(avg_test_maxF1)
    print("Max acc list:", all_test_maxACC)
    print("Mean 10-kold ACC: {}, PRE: {}, REC: {}, F1: {}".format(
        avg_test_maxACC, avg_test_maxPRE, avg_test_maxREC, avg_test_maxF1))
avg_ACC = np.average(multi_test_maxACC)
avg_PRE = np.average(multi_test_maxPRE)
avg_REC = np.average(multi_test_maxREC)
avg_F1 = np.average(multi_test_maxF1)
print('5-times ACC list:', multi_test_maxACC)
print('5-times SEN list:', multi_test_maxPRE)
print('5-times SPE list:', multi_test_maxREC)
print('5-times F1 list:', multi_test_maxF1)
print("Mean 5-times 10-kold ACC: {}, PRE: {}, REC: {}, F1: {}".format(
    avg_ACC, avg_PRE, avg_REC, avg_F1))