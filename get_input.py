# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import scipy.sparse as sp
import torch

# CATEGORY_NUMBERS = 32
CATEGORY_NUMBERS = 15
word2vec = []
with open("classFeature.csv", "r")as f:
    lines = f.readlines()
    for line in lines:
        word2vec.append([float(x) for x in line.strip().split(",")])


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cosSim(x, y):
    tmp = sum(a * b for a, b in zip(x, y))
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return round(tmp / float(non), 3)


def euDistance(x, y):
    x_center = ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)
    y_center = ((y[0] + y[2]) / 2, (y[1] + y[3]) / 2)

    distance = np.sqrt((x_center[0] - y_center[0]) ** 2 + (x_center[1] - y_center[1]) ** 2)
    return distance


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def loadData(vpath, epath, imgpath):

    assert os.path.exists(vpath)
    assert os.path.exists(imgpath)

    img = cv2.imread(imgpath)

    img_height, img_width, channel = img.shape
    features = []
    label_list = []
    image_list = []
    bbox_list=[]
    category_dict = {}


    category_list = []
    corr = np.zeros((CATEGORY_NUMBERS, CATEGORY_NUMBERS))
    adj = np.zeros((CATEGORY_NUMBERS, CATEGORY_NUMBERS))
    area_dict = {}
    area_count = 0
    with open(vpath, 'r+')as f:
        lines = f.readlines()
        for idx in range(len(lines)):
            line = lines[idx]
            if idx != 0:
                fea = list(map(float, line.strip().split(",")))
                bbox = [int(x) for x in list(map(float, line.strip().split(",")[2:]))]
                # print(bbox)
                if bbox[3] > img_height or bbox[2] > img_width:
                    continue
                if (bbox[3] - bbox[1]) <32 or (bbox[2] - bbox[0])<32:
                    continue
                bbox_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                area_dict[area_count] = ((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]), int(fea[1]))
                area_count += 1
                # print(bbox_img.shape)
                image_list.append(cv2.resize(bbox_img, (32, 32)))
                normalize_bbox = [bbox[0] / img_width, bbox[1] / img_height, bbox[2] / img_width, bbox[3] / img_height]
                bbox_list.append(normalize_bbox)
                if int(fea[1]) - 1 in category_dict.keys():
                    category_dict[int(fea[1]) - 1].append(normalize_bbox)
                else:
                    category_dict[int(fea[1]) - 1] = [normalize_bbox]
                corr[int(fea[1]) - 1, int(fea[1]) - 1] = 1
                label_list.append(int(fea[1]))
                features.append(fea[:])


    for i in range(CATEGORY_NUMBERS):
        if corr[i, i] == 1:
            for j in range(i + 1, CATEGORY_NUMBERS):
                if corr[j, j] == 1:
                    sim = cosSim(word2vec[i], word2vec[j])
                    corr[i, j] = sim
                    corr[j, i] = sim

    for i in range(CATEGORY_NUMBERS):
        category_list.append([])
    for i in range(CATEGORY_NUMBERS):
        if i in category_dict.keys():
            category_list[i] = list(np.mean(category_dict[i], axis=0)) + [len(category_dict[i])]
        else:
            category_list[i] = [0, 0, 0, 0, 0]

    for i in range(CATEGORY_NUMBERS):

        for j in range(i + 1, CATEGORY_NUMBERS):
            if i in category_dict.keys() and j in category_dict.keys():
                distance = euDistance(category_list[i], category_list[j])
                distance=1.0-distance
                # distance=1.0/distance
                # print(distance)

                adj[i, j] = distance
                adj[j, i] = distance

    img = cv2.resize(img, (128, 128))
    area_dict = sorted(area_dict.items(), key=lambda area_dict: area_dict[1][0], reverse=True)

    return image_list, label_list, bbox_list, img, adj, corr, area_dict,category_list

