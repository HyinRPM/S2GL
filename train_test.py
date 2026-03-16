import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from utils import mask_edges, lp_loss, lp_loss1
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score,precision_score,recall_score


def calculate_metric(y_true, output):
    pred_label = output.argmax(dim=1)
    acc = accuracy_score(y_true, pred_label)
    pre = precision_score(y_true, pred_label)
    rec = recall_score(y_true, pred_label)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label).ravel()
    # sen = tp / float(tp + fn)
    # spe = tn / float(tn + fp)
    f1 = f1_score(y_true, pred_label)
    return acc, pre, rec, f1

def train(model, cross_decoder, device, train_loader, optimizier, criterion, epoch, args):
    model.train()
    model.to(device)
    cross_decoder.train()
    cross_decoder.to(device)
    loss_train, n = 0, 0
    pred_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    for batch, (struc_graph, func_graph, label) in enumerate(train_loader):
        struc_graph, func_graph, label = struc_graph.to(device), func_graph.to(device), label.to(device)
        scpre_edge_index, scpos_train_edge, scedge_index = mask_edges(struc_graph, args, mask_type='dm')
        fcpre_edge_index, fcpos_train_edge, fcedge_index = mask_edges(func_graph, args, mask_type='um')
        batch_size = struc_graph.batch_size
        optimizier.zero_grad()
        output, struc_feature_list, func_feature_list = model(struc_graph, func_graph, scpre_edge_index,
                                                              fcpre_edge_index, batch_size)
        loss = criterion(output, label)

        struc_loss = lp_loss(cross_decoder, struc_feature_list, scpos_train_edge)
        func_loss = lp_loss(cross_decoder, func_feature_list, fcpos_train_edge)

        re_loss = 0
        for name, para in model.named_parameters():
            # if 'weight' in name:
            if para.requires_grad:
                re_loss = re_loss + torch.norm(para, p=2)
        loss = loss + 0.001 * re_loss


        total_loss = loss + struc_loss + func_loss

        total_loss.backward()
        optimizier.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 对模型的梯度进行裁剪，以避免梯度爆炸的问题
        # torch.nn.utils.clip_grad_norm_(cross_decoder1.parameters(), 1.0)
        # compute metric
        pred_score = torch.cat((pred_score, output), dim=0)
        labels = torch.cat((labels, label), dim=0)

        loss_train += total_loss.item()  # loss.item()返回的是loss的标量值
        n = n + 1
    acc, pre, rec, f1 = calculate_metric(labels.cpu(), pred_score.cpu())
    loss_avg = loss_train / n
    print("Epoch: {}, Train loss: {}, Train accuracy: {}, Train F1: {}".format(epoch,
                                                                 loss_avg, acc, f1))
    return loss_avg, acc, pre, rec, f1


def test(model, cross_decoder, device, test_loader, criterion, epoch, args):
    model.eval()
    cross_decoder.eval()
    loss_test, correct_test, n = 0, 0, 0
    pred_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    with torch.no_grad():
        for batch, (struc_graph, func_graph, label) in enumerate(test_loader):
            struc_graph, func_graph, label = struc_graph.to(device), func_graph.to(device), label.to(device)
            scpre_edge_index, scpos_test_edge, scedge_index = mask_edges(struc_graph, args, mask_type='dm')
            fcpre_edge_index, fcpos_test_edge, fcedge_index = mask_edges(func_graph, args, mask_type='um')
            batch_size = struc_graph.batch_size
            output, struc_feature_list, func_feature_list = model(struc_graph, func_graph, scpre_edge_index,
                                                                  fcpre_edge_index, batch_size)
            loss = criterion(output, label)
            struc_loss = lp_loss(cross_decoder, struc_feature_list, scpos_test_edge)
            func_loss = lp_loss(cross_decoder, func_feature_list, fcpos_test_edge)
            # struc_loss = lp_loss1(cross_decoder, struc_feature_list, scpos_test_edge, scedge_index, args)
            # func_loss = lp_loss1(cross_decoder, func_feature_list, fcpos_test_edge, fcedge_index, args)

            re_loss = 0
            for name, para in model.named_parameters():
                # if 'weight' in name:
                if para.requires_grad:
                    re_loss = re_loss + torch.norm(para, p=2)
            loss = loss + 0.001 * re_loss

            total_loss = loss + struc_loss + func_loss

            # compute metric
            pred_score = torch.cat((pred_score, output), dim=0)
            labels = torch.cat((labels, label), dim=0)

            loss_test += total_loss.item()  # loss.item()返回的是loss的标量值
            n = n + 1
        loss_avg = loss_test / n
        acc, pre, rec, f1 = calculate_metric(labels.cpu(), pred_score.cpu())
        print("Epoch: {}, Test loss: {}, Test accuracy: {}, Test F1: {}".format(epoch,loss_avg,
                                                                   acc, f1))
        return loss_avg, acc, pre, rec, f1
