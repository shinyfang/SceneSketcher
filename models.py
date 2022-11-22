#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from layers import GraphConvolution
from utils_model import get_network
import torch

# CATEGORY_NUMBERS = 32
CATEGORY_NUMBERS = 15
LOOP_NUM=3
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCNAttention(nn.Module):
    def __init__(self, gcn_input_shape, gcn_output_shape):
        super(GCNAttention, self).__init__()

        self.image_bbox_extract_net = get_network("inceptionv3", num_classes=2048, use_gpu=True).cuda()
        self.global_image_extract_net = get_network("inceptionv3", num_classes=CATEGORY_NUMBERS, use_gpu=True).cuda()
        # self.image_bbox_extract_net = get_network("inceptionv3", num_classes=2048, use_gpu=True)
        # self.global_image_extract_net = get_network("inceptionv3", num_classes=CATEGORY_NUMBERS, use_gpu=True)
        X = np.zeros((CATEGORY_NUMBERS, CATEGORY_NUMBERS))
        self.X = nn.Parameter(torch.from_numpy(X.astype(np.float32)))
        self.linear = nn.Linear(LOOP_NUM, 1)
        nn.init.constant_(self.X, 1e-6)
        # -----------------GCN-----------------------------
        self.gc1 = GraphConvolution(gcn_input_shape, gcn_output_shape)

    def forward(self, image_list, label_list, category_list, total_image, adj, corr):
        '''
        image_list存根据bbox截取好的图像
        label_list存根据bbox截取好的图像类别标签
        category_list存类别对应的5维输入：bbox均值,个数
        '''

        gcn_input = np.zeros((CATEGORY_NUMBERS,LOOP_NUM, 2052 ))
        category_count = np.zeros(CATEGORY_NUMBERS,dtype=np.int)
        gcn_input = Variable(torch.from_numpy(gcn_input), requires_grad=False).type(torch.FloatTensor).cuda()
        # gcn_input = Variable(torch.from_numpy(gcn_input), requires_grad=False).type(torch.FloatTensor)
        for i in range(len(label_list)):
            tmp_img = torch.from_numpy(np.transpose(np.array([image_list[i] / 255.0], np.float), [0, 3, 1, 2])).type(
                torch.FloatTensor).cuda()
            # tmp_img = torch.from_numpy(np.transpose(np.array([image_list[i] / 255.0], np.float), [0, 3, 1, 2])).type(
            #     torch.FloatTensor)
            img_feature = self.image_bbox_extract_net(tmp_img)
            if category_count[label_list[i] - 1] < LOOP_NUM:

                gcn_input[label_list[i] - 1, category_count[label_list[i] - 1], :2048] += img_feature[0]
                tmp_category = torch.from_numpy(np.array([category_list[i]], np.float)).type(torch.FloatTensor).cuda()
                # tmp_category = torch.from_numpy(np.array([category_list[i]], np.float)).type(torch.FloatTensor)
                gcn_input[label_list[i] - 1, category_count[label_list[i] - 1], 2048:] = tmp_category
                category_count[label_list[i] - 1] += 1
        gcn_input=torch.transpose(gcn_input,1,2)
        gcn_input=self.linear(gcn_input).squeeze()

        total_image = torch.from_numpy(np.transpose(np.array([total_image / 255.0], np.float), [0, 3, 1, 2])).type(
            torch.FloatTensor).cuda()
        # total_image = torch.from_numpy(np.transpose(np.array([total_image / 255.0], np.float), [0, 3, 1, 2])).type(
        #     torch.FloatTensor)
        global_attention = self.global_image_extract_net(total_image)

        corr = torch.from_numpy(corr).type(torch.FloatTensor).cuda()
        adj = torch.from_numpy(adj).type(torch.FloatTensor).cuda()
        # corr = torch.from_numpy(corr).type(torch.FloatTensor)
        # adj = torch.from_numpy(adj).type(torch.FloatTensor)
        new_adj = self.X + adj + corr

        gcn_output = F.leaky_relu(self.gc1(gcn_input, new_adj))

        result_feature = torch.mm(global_attention, gcn_output)

        return result_feature

    def get_image_feature(self, image):
        # image = torch.from_numpy(np.transpose(np.array([image / 255.0], np.float), [0, 3, 1, 2])).type(
        #     torch.FloatTensor).cuda()
        image = torch.from_numpy(np.transpose(np.array([image / 255.0], np.float), [0, 3, 1, 2])).type(
            torch.FloatTensor)
        return self.image_bbox_extract_net(image)


class TripletAttentionNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletAttentionNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, image_list_arc, label_list_arc, category_list_arc, total_image_arc, adj_arc, corr_arc,
                image_list_pos, label_list_pos, category_list_pos, total_image_pos, adj_pos, corr_pos,
                image_list_neg, label_list_neg, category_list_neg, total_image_neg, adj_neg, corr_neg
                ):
        output_pos = self.embedding_net(image_list_pos, label_list_pos, category_list_pos, total_image_pos, adj_pos,
                                        corr_pos)
        output_neg = self.embedding_net(image_list_neg, label_list_neg, category_list_neg, total_image_neg, adj_neg,
                                        corr_neg)
        output_arc = self.embedding_net(image_list_arc, label_list_arc, category_list_arc, total_image_arc, adj_arc,
                                        corr_arc)
        return output_arc, output_pos, output_neg

    def get_embedding(self, image_list, label_list, category_list, total_image, adj, corr):
        return self.embedding_net(image_list, label_list, category_list, total_image, adj, corr)

    def get_image_feature(self, image):
        return self.embedding_net.get_image_feature(image)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, adj1, x2, adj2, x3, adj3):
        output1 = self.embedding_net(x1, adj1)
        output2 = self.embedding_net(x2, adj2)
        output3 = self.embedding_net(x3, adj3)
        return output1, output2, output3

    def get_embedding(self, x, adj):
        return self.embedding_net(x, adj)


class TripletNetInception(nn.Module):
    def __init__(self):
        super(TripletNetInception, self).__init__()
        self.embedding_net = get_network("inceptionv3", num_classes=2048, use_gpu=True)

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


